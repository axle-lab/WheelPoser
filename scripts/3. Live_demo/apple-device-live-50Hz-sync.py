#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-phase WheelPoser Live Inference with UDP Data Pipeline - 50Hz Synchronized
Phase 1: Verify UDP streams and connection quality
Phase 2: Load models, calibrate, and run inference

Device Setup (ALL at 50Hz):
- pocket_phone -> Pelvis (IMU index 3)
- pocket_watch -> Left Wrist (IMU index 0)
- frame_watch -> Right Wrist (IMU index 1)
- pocket_headphone -> Head (IMU index 2)

Synchronized Architecture:
- All sensors at 50 Hz
- Fixed-length buffers (buffer_len=1 for online inference)
- Synchronized consumption similar to old IMUSet approach
- Uses FREE ACCELERATION (gravity-removed) matching Movella DOT behavior
"""
import sys

sys.path.append('.')
from pathlib import Path
import socket
import select
import numpy as np
import time
import threading
from datetime import datetime
import torch
from collections import deque
import struct

# ============ WINDOWS COMPATIBILITY ============
import platform
import pathlib

# Fix for Windows path compatibility with models saved on POSIX systems
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

# ======================= Configuration =======================

HOST = "0.0.0.0"
PORTS = [8001, 8002, 8003, 8004, 8005]
CHUNK = 8192
STATUS_INTERVAL = 1.0

# Device placement mapping to IMU indices for WheelPoser
# WheelPoser expects: [LeftWrist, RightWrist, Head, Pelvis]
STREAM_TO_IMU_INDEX = {
    "pocket_watch": 0,      # Left Wrist
    "frame_watch": 1,       # Right Wrist
    "pocket_headphone": 2,  # Head
    "pocket_phone": 3,      # Pelvis
}

# Required streams for inference (4 IMUs needed)
ACTIVE_STREAMS = ["pocket_watch", "frame_watch", "pocket_headphone", "pocket_phone"]

# Stream display names for better readability
STREAM_DISPLAY_NAMES = {
    "pocket_watch": "Left Wrist (pocket_watch)",
    "frame_watch": "Right Wrist (frame_watch)",
    "pocket_headphone": "Head (pocket_headphone)",
    "pocket_phone": "Pelvis (pocket_phone)",
}

# Unity visualizer
UNITY_VISUALIZER = False
SERVER_UNITY_IP = '127.0.0.1'
SERVER_UNITY_PORT = 8888

# ======================= Global state =======================

running = False
verification_mode = True  # Phase 1: just monitor streams
inference_mode = False     # Phase 2: run model inference
start_recording = False

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

device = get_default_device()

# ======================= Synchronized Buffer System =======================

class SynchronizedIMUBuffers:
    """
    Manages synchronized IMU buffers similar to the old IMUSet class.
    All sensors run at 50Hz, so we can use simple synchronized buffers.
    """
    def __init__(self, buffer_len=1):
        self._buffer_len = buffer_len
        self._quat_buffer = []  # List of [4, 4] arrays (4 IMUs x 4 quat components)
        self._acc_buffer = []   # List of [4, 3] arrays (4 IMUs x 3 acc components)
        
        # Individual stream buffers for incoming data
        self._stream_buffers = {stream: deque(maxlen=10) for stream in ACTIVE_STREAMS}
        self._lock = threading.Lock()
        
        # Reading state
        self._is_reading = False
        self._packet_count = 0
        self._last_packet_count = 0
        
    def add_sample(self, stream_key, quat, acc):
        """Add a new sample from UDP receiver"""
        with self._lock:
            self._stream_buffers[stream_key].append((quat, acc))
            
    def _try_consume_synchronized_sample(self):
        """
        Try to consume one sample from each stream to create a synchronized measurement.
        Similar to the old script's approach where all IMUs are read together.
        """
        # Check if all streams have at least one sample
        if not all(len(self._stream_buffers[s]) > 0 for s in ACTIVE_STREAMS):
            return False
        
        # Consume oldest sample from each stream (FIFO)
        quats = [None] * 4
        accs = [None] * 4
        
        for stream in ACTIVE_STREAMS:
            imu_idx = STREAM_TO_IMU_INDEX[stream]
            quat, acc = self._stream_buffers[stream].popleft()
            quats[imu_idx] = quat
            accs[imu_idx] = acc
        
        # Create synchronized measurement
        full_measurement = np.array([quats, accs])  # [2, 4, components]
        
        # Add to rolling buffer (similar to old script's truncation)
        truncate = int(len(self._quat_buffer) == self._buffer_len)
        self._quat_buffer = self._quat_buffer[truncate:] + [np.array(quats, dtype=float)]
        self._acc_buffer = self._acc_buffer[truncate:] + [np.array(accs, dtype=float)]
        
        self._packet_count += 1
        return True
        
    def start_reading(self):
        """Start consuming samples into synchronized buffers"""
        with self._lock:
            self._is_reading = True
            self._quat_buffer = []
            self._acc_buffer = []
            
    def stop_reading(self):
        """Stop consuming samples"""
        with self._lock:
            self._is_reading = False
            
    def clear_buffer(self):
        """Clear the synchronized buffers"""
        with self._lock:
            self._quat_buffer = []
            self._acc_buffer = []
            
    def update(self):
        """
        Try to consume synchronized samples.
        Should be called regularly (e.g., in a loop or timer).
        """
        with self._lock:
            if not self._is_reading:
                return
                
            # Try to consume as many synchronized samples as possible
            while self._try_consume_synchronized_sample():
                pass
                
    def get_current_buffer(self):
        """
        Get current buffer as torch tensors.
        Returns: (orientations, accelerations) as [buffer_len, 4, 4/3]
        """
        with self._lock:
            if len(self._quat_buffer) == 0 or len(self._acc_buffer) == 0:
                return torch.tensor([]), torch.tensor([])
                
            q = torch.tensor(np.array(self._quat_buffer), dtype=torch.float32)
            a = torch.tensor(np.array(self._acc_buffer), dtype=torch.float32)
            return q, a
            
    def has_new_data(self):
        """Check if new data has been consumed since last check"""
        with self._lock:
            has_new = self._packet_count != self._last_packet_count
            self._last_packet_count = self._packet_count
            return has_new
            
    def get_packet_count(self):
        """Get current packet count"""
        with self._lock:
            return self._packet_count

# Global buffer instance
imu_buffers = SynchronizedIMUBuffers(buffer_len=1)

# Calibration matrices (will be set during calibration)
smpl2imu = None
device2bone = None
acc_offsets = None

# Recording
is_recording = False
record_buffer = None
record_session_start = 0.0

# ======================= FPS tracking =======================

class FPSMeter:
    def __init__(self, horizon_sec=2.0, maxlen=200):
        self.ts = deque(maxlen=maxlen)
        self.horizon = float(horizon_sec)

    def update(self, now: float):
        self.ts.append(now)

    def get_fps(self, now: float) -> float:
        horizon_start = now - self.horizon
        while self.ts and self.ts[0] < horizon_start:
            self.ts.popleft()
        if len(self.ts) < 2:
            return 0.0
        dur = self.ts[-1] - self.ts[0]
        return (len(self.ts) - 1) / dur if dur > 0 else 0.0

# FPS meters for required streams only
fps_meters = {stream: FPSMeter() for stream in ACTIVE_STREAMS}
inference_fps = FPSMeter()

# ======================= UDP parsing =======================

def parse_udp_payload(payload: bytes):
    """
    Parse UDP payload to extract IMU data.
    Expected format: "<placement>;<type>:\n<rows...>"
    Each row: unix_timestamp sensor_timestamp gx gy gz fax fay faz qx qy qz qw avx avy avz
    
    Note: 'free_acc' (fax, fay, faz) is acceleration with gravity already removed by the sensor.
          This matches the Movella DOT behavior where freeAcceleration() excludes gravity.
    
    Returns: list of (stream_key, unix_ts, sensor_ts, gravity, free_acc, quat, gyro)
    """
    try:
        msg = payload.decode("utf-8").strip()
    except Exception:
        return []

    if ';' not in msg or ':' not in msg:
        return []

    try:
        placement, rhs = msg.split(";", 1)
        placement = placement.strip()
        dtype, body = rhs.split(":", 1)
        dtype = dtype.strip()
        body = body.strip()
    except ValueError:
        return []

    if dtype not in ["phone", "watch", "headphone"]:
        return []

    events = []
    for line in body.splitlines():
        fields = line.strip().split()
        if len(fields) != 15:  # unix_ts, sensor_ts, 3 gravity, 3 free_acc, 4 quat, 3 gyro
            continue
        
        try:
            unix_ts = float(fields[0])
            sensor_ts = float(fields[1])
            gravity = [float(fields[2]), float(fields[3]), float(fields[4])]
            free_acc = [float(fields[5]), float(fields[6]), float(fields[7])]
            quat = [float(fields[8]), float(fields[9]), float(fields[10]), float(fields[11])]
            gyro = [float(fields[12]), float(fields[13]), float(fields[14])]
            
            stream_key = f"{placement}_{dtype}"
            
            # Only process streams we care about
            if stream_key in ACTIVE_STREAMS:
                events.append((stream_key, unix_ts, sensor_ts, gravity, free_acc, quat, gyro))
        except ValueError:
            continue

    return events

# ======================= UDP receiver thread =======================

def udp_receiver_thread(sockets):
    """Continuously receive UDP packets and add to stream buffers"""
    global running
    
    empty = []
    while running:
        readable, _, _ = select.select(sockets, empty, empty, 0.1)
        
        for s in readable:
            try:
                data, _ = s.recvfrom(CHUNK)
            except Exception:
                continue
            
            events = parse_udp_payload(data)
            for stream_key, unix_ts, sensor_ts, gravity, free_acc, quat, gyro in events:
                now = time.time()
                
                # Update FPS for this stream
                if stream_key in fps_meters:
                    fps_meters[stream_key].update(now)
                
                # Convert quaternion from [qx, qy, qz, qw] to [qw, qx, qy, qz]
                quat_wxyz = [quat[3], quat[0], quat[1], quat[2]]
                
                # Use free acceleration (gravity already removed by sensor)
                # This matches the old Movella DOT script behavior
                free_acceleration = free_acc
                
                # Add to synchronized buffer system
                imu_buffers.add_sample(stream_key, quat_wxyz, free_acceleration)

def buffer_update_thread():
    """Background thread to consume samples into synchronized buffers"""
    global running
    
    while running:
        imu_buffers.update()
        time.sleep(0.001)  # Check frequently for new data

# ======================= Stream verification display =======================

def display_stream_status():
    """Display real-time stream status during verification phase"""
    print("\n" + "="*80)
    print("STREAM VERIFICATION MODE")
    print("="*80)
    print("\nMonitoring UDP streams. Press 'c' when all 4 devices are streaming.")
    print("\nRequired Device Setup (ALL at 50Hz):")
    for stream in ACTIVE_STREAMS:
        imu_idx = STREAM_TO_IMU_INDEX[stream]
        print(f"  - {STREAM_DISPLAY_NAMES[stream]} (IMU index {imu_idx})")
    print("\nPress 'q' to quit\n")
    
    last_print = time.time()
    
    while verification_mode and running:
        now = time.time()
        
        if now - last_print > STATUS_INTERVAL:
            # Build status display
            lines = []
            lines.append("\n" + "-"*80)
            lines.append(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            lines.append("-"*80)
            
            # Show all required streams in anatomical order
            anatomical_order = ["pocket_watch", "frame_watch", "pocket_headphone", "pocket_phone"]
            
            all_active = True
            for stream in anatomical_order:
                fps = fps_meters[stream].get_fps(now)
                is_active = fps > 10  # Consider active if getting data
                imu_idx = STREAM_TO_IMU_INDEX[stream]
                
                if not is_active:
                    all_active = False
                
                # Status indicator
                status = "✓ ACTIVE " if is_active else "✗ MISSING"
                
                # FPS display
                fps_str = f"{fps:6.1f} Hz" if fps > 0 else "   0.0 Hz"
                
                # Build line with display name
                display_name = STREAM_DISPLAY_NAMES[stream]
                line = f"  {status} | {display_name:35s} | {fps_str} | IMU[{imu_idx}]"
                
                lines.append(line)
            
            # Summary
            lines.append("-"*80)
            packet_count = imu_buffers.get_packet_count()
            lines.append(f"Synchronized packets consumed: {packet_count}")
            
            if all_active:
                lines.append("✓ ALL DEVICES ACTIVE - Ready to proceed!")
                lines.append("  Press 'c' to continue to model loading and calibration")
            else:
                lines.append("⚠ Waiting for all devices to stream...")
            
            lines.append("-"*80)
            
            # Cross-platform terminal clear
            if platform.system() == 'Windows':
                import os
                os.system('cls')
            else:
                print("\033[2J\033[H", end="")
            
            print("\n".join(lines), flush=True)
            
            last_print = now
        
        time.sleep(0.1)

# ======================= Calibration =======================

def get_mean_measurement_of_n_second(num_seconds=3, buffer_len=150):
    """
    Collect IMU data for num_seconds and return the average of measurements.
    Mimics the old IMUSet.get_mean_measurement_of_n_second function.
    
    Args:
        num_seconds: How many seconds to collect data
        buffer_len: Buffer length for collection
    
    Returns:
        Mean quaternion and acceleration torch.Tensor in shape [4, 4] and [4, 3] respectively.
    """
    print(f'Collecting {num_seconds} seconds of calibration data...')
    
    # Temporarily use larger buffer for calibration
    old_buffer_len = imu_buffers._buffer_len
    imu_buffers._buffer_len = buffer_len
    
    # Clear and start reading
    imu_buffers.clear_buffer()
    imu_buffers.start_reading()
    
    # Wait for collection
    time.sleep(num_seconds)
    
    # Stop reading
    imu_buffers.stop_reading()
    
    # Get collected data
    q, a = imu_buffers.get_current_buffer()
    
    # Restore buffer length
    imu_buffers._buffer_len = old_buffer_len
    
    if q.size(0) == 0:
        raise RuntimeError("No IMU data received during calibration!")
    
    actual_samples = q.size(0)
    actual_rate = actual_samples / num_seconds
    expected_rate = 50  # All sensors at 50Hz
    
    print(f"Collected {actual_samples} samples over {num_seconds} seconds ({actual_rate:.1f} Hz)")
    
    if actual_rate < expected_rate * 0.7:
        print(f"  ⚠ Warning: Sample rate is lower than expected ({expected_rate} Hz)")
    
    # Average the samples
    oris = q.mean(dim=0)  # [4, 4] - quaternions for each IMU
    accs = a.mean(dim=0)  # [4, 3] - accelerations for each IMU
    
    return oris, accs

def perform_reference_frame_calibration():
    """
    STAGE 1: Establish the global reference frame using Left Wrist sensor.
    This computes the smpl2imu rotation matrix that transforms from SMPL coordinate system
    to the sensor's inertial frame.
    
    Returns:
        smpl2imu: [3, 3] rotation matrix (global to sensor frame)
    """
    from src.articulate.math import quaternion_to_rotation_matrix
    
    print('\n[Stage 1/2] Reference Frame Calibration')
    print('='*80)
    input('Place Left Wrist sensor (pocket_watch) aligned with body reference frame:\n'
          '  - x = Left\n'
          '  - y = Up\n'
          '  - z = Forward\n'
          'Press Enter when ready.')
    
    for i in range(3, 0, -1):
        print(f'\rHold steady... {i}s', end='', flush=True)
        time.sleep(1)
    print()
    
    # Collect averaged orientation from all sensors (use Left Wrist only)
    oris, _ = get_mean_measurement_of_n_second(num_seconds=3, buffer_len=200)
    
    # Use only the Left Wrist (index 0) to establish reference frame
    left_wrist_quat = oris[0]  # [4] quaternion
    smpl2imu = quaternion_to_rotation_matrix(left_wrist_quat).view(3, 3).t()  # global to sensor frame
    
    print('✓ Reference frame established')
    print(f'  smpl2imu rotation matrix computed from Left Wrist orientation')
    
    return smpl2imu

def perform_tpose_calibration(smpl2imu):
    """
    STAGE 2: Calibrate sensor-to-bone transformations and acceleration offsets in T-pose.
    This uses the reference frame from Stage 1 to compute device2bone rotations
    and acceleration offsets for all sensors.
    
    Args:
        smpl2imu: [3, 3] rotation matrix from Stage 1
    
    Returns:
        device2bone: [4, 3, 3] rotation matrices for each sensor
        acc_offsets: [4, 3, 1] acceleration offsets for each sensor
    """
    from src.articulate.math import quaternion_to_rotation_matrix
    
    print('\n[Stage 2/2] T-Pose Calibration')
    print('='*80)
    input('Wear all 4 IMUs correctly and stand in T-pose.\n'
          'Press Enter when ready.')
    
    for i in range(3, 0, -1):
        print(f'\rHold T-pose... {i}s', end='', flush=True)
        time.sleep(1)
    print()
    
    # Collect averaged measurements from all sensors
    oris, accs = get_mean_measurement_of_n_second(num_seconds=3, buffer_len=200)
    
    # Convert quaternions to rotation matrices
    oris_mat = quaternion_to_rotation_matrix(oris)  # [4, 3, 3] - sensor to global
    
    # Compute device2bone: transforms from device frame to bone frame
    device2bone = smpl2imu.matmul(oris_mat).transpose(1, 2).matmul(torch.eye(3))
    
    # Compute acceleration offsets in global inertial frame
    acc_offsets = smpl2imu.matmul(accs.unsqueeze(-1))  # [4, 3, 1]
    
    print('✓ T-pose calibration complete')
    print(f'  device2bone matrices computed for all 4 sensors')
    print(f'  Acceleration offsets computed')
    
    return device2bone, acc_offsets

# ======================= Model loading =======================

def load_wheelposer_models():
    """Load all WheelPoser models (Phase 2 only)"""
    print("\n" + "="*80)
    print("LOADING WHEELPOSER MODELS")
    print("="*80)
    
    # Import all model dependencies (only when needed)
    from src.config import Config, joint_set
    from src.models.utils import get_model
    from src.utils import get_checkpoints
    from src.models.LSTMs.Three_Stage_Global.Three_Stage_Global_WheelPoser_Wrapper import Three_Stage_Global_WheelPoser
    
    num_past_frame = 20
    num_future_frame = 5
    physics = False
    
    # Load AMASS models
    print("\n[1/3] Loading AMASS base models...")
    model_names = ["IMU2Leaf_WheelPoser_AMASS", "Leaf2Full_WheelPoser_AMASS", "Full2Pose_WheelPoser_AMASS"]
    experiment_names = "TransPose_Style_500"
    leave_one_out = 'am_only'
    
    best_ckpts = get_checkpoints(model_names, experiment_names, leave_one_out=leave_one_out)
    
    AMASS_IMU2Leaf_config = Config(experiment=experiment_names, model=model_names[0], project_root_dir=".", 
                                   joints_set=joint_set.WheelPoser, pred_joints_set=joint_set.upper_body,
                                   normalize=True, r6d=True, loss_type="mse", use_joint_loss=False, 
                                   mkdir=False, upper_body_only=True)
    AMASS_IMU2Leaf_model = get_model(AMASS_IMU2Leaf_config).load_from_checkpoint(
        best_ckpts[model_names[0]], config=AMASS_IMU2Leaf_config)
    
    AMASS_Leaf2Full_config = Config(experiment=experiment_names, model=model_names[1], project_root_dir=".", 
                                    joints_set=joint_set.WheelPoser, pred_joints_set=joint_set.upper_body,
                                    normalize=True, r6d=True, loss_type="mse", use_joint_loss=False, 
                                    mkdir=False, upper_body_only=True)
    AMASS_Leaf2Full_model = get_model(AMASS_Leaf2Full_config).load_from_checkpoint(
        best_ckpts[model_names[1]], config=AMASS_Leaf2Full_config)
    
    AMASS_Full2Pose_config = Config(experiment=experiment_names, model=model_names[2], project_root_dir=".", 
                                    joints_set=joint_set.WheelPoser, pred_joints_set=joint_set.upper_body,
                                    normalize=True, r6d=True, loss_type="mse", use_joint_loss=False, 
                                    mkdir=False, upper_body_only=True)
    AMASS_Full2Pose_model = get_model(AMASS_Full2Pose_config).load_from_checkpoint(
        best_ckpts[model_names[2]], config=AMASS_Full2Pose_config)
    
    print("✓ AMASS models loaded")
    
    # Load WheelPoser models
    print("\n[2/3] Loading WheelPoser fine-tuned models...")
    model_names = ["IMU2Leaf_WheelPoser_WHEELPOSER", "Leaf2Full_WheelPoser_WHEELPOSER", "Full2Pose_WheelPoser_WHEELPOSER"]
    best_ckpts = get_checkpoints(model_names, experiment_names, leave_one_out=leave_one_out)
    
    WHEELPOSER_IMU2Leaf_config = Config(experiment=experiment_names, model=model_names[0], project_root_dir=".", 
                                        joints_set=joint_set.WheelPoser, pred_joints_set=joint_set.upper_body,
                                        normalize=True, r6d=True, loss_type="mse", use_joint_loss=False, 
                                        mkdir=False, upper_body_only=True, exp_setup='am_only', upsample_copies=7)
    WHEELPOSER_IMU2Leaf_model = get_model(WHEELPOSER_IMU2Leaf_config, pretrained=AMASS_IMU2Leaf_model).load_from_checkpoint(
        best_ckpts[model_names[0]], config=WHEELPOSER_IMU2Leaf_config, pretrained_model=AMASS_IMU2Leaf_model)
    
    WHEELPOSER_Leaf2Full_config = Config(experiment=experiment_names, model=model_names[1], project_root_dir=".", 
                                         joints_set=joint_set.WheelPoser, pred_joints_set=joint_set.upper_body,
                                         normalize=True, r6d=True, loss_type="mse", use_joint_loss=False, 
                                         mkdir=False, upper_body_only=True, exp_setup='am_only', upsample_copies=7)
    WHEELPOSER_Leaf2Full_model = get_model(WHEELPOSER_Leaf2Full_config, pretrained=AMASS_Leaf2Full_model).load_from_checkpoint(
        best_ckpts[model_names[1]], config=WHEELPOSER_Leaf2Full_config, pretrained_model=AMASS_Leaf2Full_model)
    
    WHEELPOSER_Full2Pose_config = Config(experiment=experiment_names, model=model_names[2], project_root_dir=".", 
                                         joints_set=joint_set.WheelPoser, pred_joints_set=joint_set.upper_body,
                                         normalize=True, r6d=True, loss_type="mse", use_joint_loss=False, 
                                         mkdir=False, upper_body_only=True, exp_setup='am_only', upsample_copies=7)
    WHEELPOSER_Full2Pose_model = get_model(WHEELPOSER_Full2Pose_config, pretrained=AMASS_Full2Pose_model).load_from_checkpoint(
        best_ckpts[model_names[2]], config=WHEELPOSER_Full2Pose_config, pretrained_model=AMASS_Full2Pose_model)
    
    print("✓ WheelPoser models loaded")
    
    # Create WheelPoser pipeline
    print("\n[3/3] Building inference pipeline...")
    shared_input_config = WHEELPOSER_IMU2Leaf_config
    wheelposer = Three_Stage_Global_WheelPoser(
        config=shared_input_config, 
        imu2leaf=WHEELPOSER_IMU2Leaf_model, 
        leaf2full=WHEELPOSER_Leaf2Full_model, 
        full2pose=WHEELPOSER_Full2Pose_model, 
        num_past_frame=num_past_frame, 
        num_future_frame=num_future_frame, 
        physics=physics
    ).to(device)
    wheelposer.eval()
    
    print(f"✓ Pipeline ready on device: {device}")
    print("="*80)
    
    return wheelposer, WHEELPOSER_Full2Pose_config

# ======================= Keyboard control =======================

def input_thread():
    global running, start_recording, verification_mode, inference_mode
    
    while running:
        try:
            c = input().strip().lower()
        except EOFError:
            break
        
        if c == 'q':
            running = False
            verification_mode = False
            inference_mode = False
        elif c == 'c' and verification_mode:
            # Check if all required streams are active
            all_active = all(fps_meters[s].get_fps(time.time()) > 10 for s in ACTIVE_STREAMS)
            
            if all_active:
                verification_mode = False
                print("\n✓ Proceeding to model loading phase...")
            else:
                print(f"\n⚠ Cannot proceed - not all devices are streaming at sufficient rate")
        elif c == 'r' and inference_mode:
            start_recording = True
            print("\n[REC] Recording started")
        elif c == 's' and inference_mode:
            start_recording = False
            print("\n[REC] Recording stopped")

# ======================= Inference loop =======================

def run_inference(wheelposer, config):
    """Run the main inference loop (Phase 2) - Synchronized 50Hz"""
    global inference_mode, is_recording, record_buffer, record_session_start, start_recording
    
    from src.articulate.math import quaternion_to_rotation_matrix
    import pygame
    
    inference_mode = True
    
    # Setup Unity (optional)
    conn = None
    if UNITY_VISUALIZER:
        print("\nSetting up Unity visualizer...")
        server_for_unity = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_for_unity.bind((SERVER_UNITY_IP, SERVER_UNITY_PORT))
        server_for_unity.listen(5)
        print(f'Waiting for Unity to connect on {SERVER_UNITY_IP}:{SERVER_UNITY_PORT}...')
        print('(Press Ctrl+C to skip Unity and continue)')
        
        try:
            server_for_unity.settimeout(60.0)
            conn, addr_unity = server_for_unity.accept()
            print(f'✓ Unity connected from {addr_unity}')
        except socket.timeout:
            print('⚠ Unity connection timeout - continuing without visualization')
            conn = None
        except KeyboardInterrupt:
            print('\n⚠ Skipping Unity - continuing without visualization')
            conn = None
    
    print("\n" + "="*80)
    print("LIVE INFERENCE MODE - Synchronized 50Hz")
    print("="*80)
    print("All sensors at 50Hz, synchronized buffer consumption")
    print("Controls: [r] start recording | [s] stop recording | [q] quit")
    print()
    
    # Start reading from synchronized buffers
    imu_buffers.start_reading()
    
    # Statistics
    frames_processed = 0
    old_packet_count = 0
    
    pygame.init()
    last_status_time = time.time()
    
    # Warmup
    print("Warming up...")
    time.sleep(0.5)
    print("✓ Starting inference")
    
    try:
        while inference_mode and running:
            # Check if new data is available (similar to old script's packet_count check)
            current_packet_count = imu_buffers.get_packet_count()
            
            if current_packet_count == old_packet_count:
                # No new data, wait a bit
                time.sleep(0.001)
                continue
            
            old_packet_count = current_packet_count
            
            # Get current buffer (similar to old script's get_current_buffer)
            ori_raw, acc_raw = imu_buffers.get_current_buffer()
            
            if ori_raw.size(0) == 0 or acc_raw.size(0) == 0:
                continue
            
            frames_processed += 1
            
            # Move to device
            ori_raw = ori_raw.to(device)
            acc_raw = acc_raw.to(device)
            
            # Calibrate (same as old script)
            ori_raw = quaternion_to_rotation_matrix(ori_raw).view(1, 4, 3, 3)
            acc_cal = (smpl2imu.matmul(acc_raw.view(-1, 4, 3, 1)) - acc_offsets).view(1, 4, 3)
            ori_cal = smpl2imu.matmul(ori_raw).matmul(device2bone)
            imu_recording = torch.cat((acc_cal.view(-1, 12), ori_cal.view(-1, 36)), dim=1)
            
            # Normalize for model (same as old script)
            acc = torch.cat((acc_cal[:, :3] - acc_cal[:, 3:], acc_cal[:, 3:]), dim=1).bmm(ori_cal[:, -1]) / config.acc_scale
            ori = torch.cat((ori_cal[:, 3:].transpose(2, 3).matmul(ori_cal[:, :3]), ori_cal[:, 3:]), dim=1)
            data_nn = torch.cat((acc.view(-1, 12), ori.view(-1, 36)), dim=1)
            
            # Run inference
            with torch.no_grad():
                pose = wheelposer.forward_online(data_nn)
            tran = torch.tensor([0, -0.4, -0.1055]).to(device)
            
            # Update FPS
            inference_fps.update(time.time())
            
            # Recording (same as old script)
            if not is_recording and start_recording:
                record_buffer = imu_recording.view(1, -1)
                is_recording = True
                record_session_start = time.time()
            elif is_recording and start_recording:
                record_buffer = torch.cat([record_buffer, imu_recording.view(1, -1)], dim=0)
            elif is_recording and not start_recording:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = Path('src/data/imu_recordings')
                save_path.mkdir(exist_ok=True, parents=True)
                torch.save(record_buffer, save_path / f'r{timestamp}.pt')
                recording_fps = record_buffer.size(0) / (time.time() - record_session_start)
                print(f'\n[REC] Saved {record_buffer.size(0)} frames at {recording_fps:.1f} FPS')
                is_recording = False
            
            # Send to Unity
            if conn:
                s = ','.join(['%g' % v for v in pose]) + '#' + \
                    ','.join(['%g' % v for v in tran]) + '$'
                try:
                    conn.send(s.encode('utf8'))
                except:
                    pass
            
            # Status print
            now = time.time()
            if now - last_status_time > STATUS_INTERVAL:
                status_parts = []
                for stream in ["pocket_watch", "frame_watch", "pocket_headphone", "pocket_phone"]:
                    short_name = stream.replace("pocket_", "P-").replace("frame_", "F-").replace("phone", "Ph").replace("watch", "W").replace("headphone", "H")
                    fps_val = fps_meters[stream].get_fps(now)
                    status_parts.append(f"{short_name}:{fps_val:4.1f}")
                
                stream_status = " | ".join(status_parts)
                inf_fps = inference_fps.get_fps(now)
                rec_status = "●REC" if is_recording else "○---"
                
                print(f"\r[{rec_status}] {stream_status} | Inf:{inf_fps:5.1f} Hz | Frames:{frames_processed}", 
                      end="", flush=True)
                last_status_time = now
    
    except KeyboardInterrupt:
        print("\n\nInference interrupted by user")
    finally:
        if conn:
            try:
                conn.close()
            except:
                pass
        
        imu_buffers.stop_reading()
        
        # Print final statistics
        if frames_processed > 0:
            print(f"\n\n{'='*80}")
            print("FINAL STATISTICS")
            print('='*80)
            print(f"Frames processed: {frames_processed}")
            print('='*80)

# ======================= Main =======================

def main():
    global running, verification_mode, smpl2imu, device2bone, acc_offsets
    
    print("\n" + "="*80)
    print("WHEELPOSER LIVE INFERENCE - SYNCHRONIZED 50Hz PIPELINE")
    print("="*80)
    print(f"\nPlatform: {platform.system()}")
    print(f"Device: {device}")
    print("\nDevice Configuration:")
    print("  - Left Wrist:  pocket_watch (50 Hz)")
    print("  - Right Wrist: frame_watch (50 Hz)")
    print("  - Head:        pocket_headphone (50 Hz)")
    print("  - Pelvis:      pocket_phone (50 Hz)")
    print("\nArchitecture:")
    print("  - All sensors synchronized at 50 Hz")
    print("  - Fixed-length buffers with FIFO consumption")
    print("  - Similar to original IMUSet approach")
    print("\nPhase 1: Stream Verification")
    print("Phase 2: Model Loading → Calibration → Inference")
    print()
    
    # ===== Setup UDP =====
    print("Setting up UDP receivers...")
    sockets = []
    for p in PORTS:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # SO_REUSEPORT doesn't exist on Windows
        if platform.system() != 'Windows':
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except (AttributeError, OSError):
                pass
        
        sock.bind((HOST, p))
        sockets.append(sock)
    print(f"✓ Listening on ports: {PORTS}")
    
    # Start UDP receiver thread
    running = True
    verification_mode = True
    udp_thread = threading.Thread(target=udp_receiver_thread, args=(sockets,), daemon=True)
    udp_thread.start()
    
    # Start buffer update thread
    buffer_thread = threading.Thread(target=buffer_update_thread, daemon=True)
    buffer_thread.start()
    
    # Start keyboard input thread
    input_thread_obj = threading.Thread(target=input_thread, daemon=True)
    input_thread_obj.start()
    
    # ===== PHASE 1: Stream Verification =====
    display_thread = threading.Thread(target=display_stream_status, daemon=True)
    display_thread.start()
    
    # Wait for user to confirm streams are good
    while verification_mode and running:
        time.sleep(0.1)
    
    if not running:
        print("\nExiting...")
        running = False
        time.sleep(0.5)
        for s in sockets:
            try:
                s.close()
            except:
                pass
        return
    
    # ===== PHASE 2: Model Loading =====
    try:
        wheelposer, config = load_wheelposer_models()
    except Exception as e:
        print(f"\n✗ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        print("Exiting...")
        running = False
        time.sleep(0.5)
        for s in sockets:
            try:
                s.close()
            except:
                pass
        return
    
    # ===== PHASE 2: Calibration =====
    print("\n" + "="*80)
    print("CALIBRATION")
    print("="*80)
    print("\nTwo-stage calibration process:")
    print("  Stage 1: Establish global reference frame (Left Wrist sensor)")
    print("  Stage 2: Calibrate all sensors in T-pose")
    
    try:
        # Stage 1: Reference frame calibration
        smpl2imu = perform_reference_frame_calibration()
        
        # Stage 2: T-pose calibration
        device2bone, acc_offsets = perform_tpose_calibration(smpl2imu)
        
        # Move to device
        smpl2imu = smpl2imu.to(device)
        device2bone = device2bone.to(device)
        acc_offsets = acc_offsets.to(device)
        
        print("\n" + "="*80)
        print("✓ CALIBRATION COMPLETE")
        print("="*80)
        print(f"  smpl2imu shape: {smpl2imu.shape}")
        print(f"  device2bone shape: {device2bone.shape}")
        print(f"  acc_offsets shape: {acc_offsets.shape}")
        
        # ===== PHASE 2: Inference =====
        run_inference(wheelposer, config)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except EOFError:
        print("\n\nInput ended")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean shutdown
        print("\nShutting down...")
        running = False
        verification_mode = False
        inference_mode = False
        
        time.sleep(0.5)
        
        for s in sockets:
            try:
                s.close()
            except Exception:
                pass
        
        print("Shutdown complete")

if __name__ == "__main__":
    main()