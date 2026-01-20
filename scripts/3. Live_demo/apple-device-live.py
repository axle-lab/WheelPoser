#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-phase WheelPoser Live Inference with UDP Data Pipeline
Phase 1: Verify UDP streams and connection quality
Phase 2: Load models, calibrate, and run inference

Device Setup:
- pocket_phone -> Pelvis (IMU index 3)
- pocket_watch -> Left Wrist (IMU index 0)
- frame_watch -> Right Wrist (IMU index 1)
- pocket_headphone -> Head (IMU index 2)
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

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# IMU data buffers (per stream)
latest_imu_data = {}  # stream_key -> {'quat': [qx,qy,qz,qw], 'acc': [ax,ay,az], 'timestamp': float}
data_lock = threading.Lock()

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
    """Continuously receive UDP packets and update latest_imu_data"""
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
                
                # Total acceleration = gravity + free_acc
                total_acc = [gravity[i] + free_acc[i] for i in range(3)]
                
                with data_lock:
                    latest_imu_data[stream_key] = {
                        'quat': quat_wxyz,
                        'acc': total_acc,
                        'timestamp': unix_ts
                    }

# ======================= Stream verification display =======================

def display_stream_status():
    """Display real-time stream status during verification phase"""
    print("\n" + "="*80)
    print("STREAM VERIFICATION MODE")
    print("="*80)
    print("\nMonitoring UDP streams. Press 'c' when all 4 devices are streaming.")
    print("\nRequired Device Setup:")
    for stream in ACTIVE_STREAMS:
        imu_idx = STREAM_TO_IMU_INDEX[stream]
        print(f"  - {STREAM_DISPLAY_NAMES[stream]} (IMU index {imu_idx})")
    print("\nPress 'q' to quit\n")
    
    last_print = time.time()
    
    while verification_mode and running:
        now = time.time()
        
        if now - last_print > STATUS_INTERVAL:
            # Get current stream status
            with data_lock:
                active_streams = list(latest_imu_data.keys())
            
            # Build status display
            lines = []
            lines.append("\n" + "-"*80)
            lines.append(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            lines.append("-"*80)
            
            # Show all required streams in anatomical order
            anatomical_order = ["pocket_watch", "frame_watch", "pocket_headphone", "pocket_phone"]
            
            for stream in anatomical_order:
                fps = fps_meters[stream].get_fps(now)
                is_active = stream in active_streams
                imu_idx = STREAM_TO_IMU_INDEX[stream]
                
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
            required_active = sum(1 for s in ACTIVE_STREAMS if s in active_streams)
            total_required = len(ACTIVE_STREAMS)
            lines.append(f"Status: {required_active}/{total_required} devices streaming")
            
            if required_active == total_required:
                lines.append("✓ ALL DEVICES ACTIVE - Ready to proceed!")
                lines.append("  Press 'c' to continue to model loading and calibration")
            else:
                missing = [STREAM_DISPLAY_NAMES[s] for s in ACTIVE_STREAMS if s not in active_streams]
                lines.append(f"⚠ Missing: {', '.join(missing)}")
            
            lines.append("-"*80)
            
            # Clear and print
            print("\033[2J\033[H" + "\n".join(lines), flush=True)
            
            last_print = now
        
        time.sleep(0.1)

# ======================= IMU data aggregation =======================

def get_current_imu_measurement():
    """
    Aggregate latest IMU data from all active streams.
    Returns data in WheelPoser order: [LeftWrist, RightWrist, Head, Pelvis]
    Returns: (orientations, accelerations) as torch tensors [1, 4, 4/3]
    """
    with data_lock:
        # Check all required streams have data
        if not all(s in latest_imu_data for s in ACTIVE_STREAMS):
            return None, None
        
        # Build arrays in WheelPoser order using STREAM_TO_IMU_INDEX mapping
        # Create array indexed by IMU position
        quats = [None] * 4
        accs = [None] * 4
        
        for stream in ACTIVE_STREAMS:
            imu_idx = STREAM_TO_IMU_INDEX[stream]
            data = latest_imu_data[stream]
            quats[imu_idx] = data['quat']
            accs[imu_idx] = data['acc']
        
        # Verify no None values (shouldn't happen if check above passes)
        if None in quats or None in accs:
            return None, None
        
        # Convert to torch tensors
        ori_tensor = torch.tensor([quats], dtype=torch.float32)  # [1, 4, 4]
        acc_tensor = torch.tensor([accs], dtype=torch.float32)   # [1, 4, 3]
        
        return ori_tensor, acc_tensor

# ======================= Calibration =======================

def perform_calibration(wait_seconds=3):
    """
    Perform calibration by collecting samples over wait_seconds.
    Returns calibration matrices.
    """
    print(f'\nCollecting {wait_seconds} seconds of calibration data...')
    
    # Collect samples
    quat_samples = []
    acc_samples = []
    start = time.time()
    sample_count = 0
    
    while time.time() - start < wait_seconds:
        ori, acc = get_current_imu_measurement()
        if ori is not None and acc is not None:
            quat_samples.append(ori)
            acc_samples.append(acc)
            sample_count += 1
        time.sleep(0.01)
    
    if len(quat_samples) == 0:
        raise RuntimeError("No IMU data received during calibration!")
    
    print(f"Collected {sample_count} samples")
    
    # Average the samples
    oris = torch.cat(quat_samples, dim=0).mean(dim=0)  # [4, 4]
    accs = torch.cat(acc_samples, dim=0).mean(dim=0)   # [4, 3]
    
    # Import needed functions (lazy import for phase 2)
    from src.articulate.math import quaternion_to_rotation_matrix
    
    # Compute calibration for sensor 0 (LeftWrist as reference)
    smpl2imu_val = quaternion_to_rotation_matrix(oris[0]).view(3, 3).t()
    
    # Compute device2bone for all sensors
    oris_mat = quaternion_to_rotation_matrix(oris)  # [4, 3, 3]
    device2bone_val = smpl2imu_val.matmul(oris_mat).transpose(1, 2).matmul(torch.eye(3))
    
    # Compute acceleration offsets
    acc_offsets_val = smpl2imu_val.matmul(accs.unsqueeze(-1))  # [4, 3, 1]
    
    return smpl2imu_val, device2bone_val, acc_offsets_val

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
            with data_lock:
                active = list(latest_imu_data.keys())
            
            if all(s in active for s in ACTIVE_STREAMS):
                verification_mode = False
                print("\n✓ Proceeding to model loading phase...")
            else:
                missing = [STREAM_DISPLAY_NAMES[s] for s in ACTIVE_STREAMS if s not in active]
                print(f"\n⚠ Cannot proceed - missing devices: {', '.join(missing)}")
        elif c == 'r' and inference_mode:
            start_recording = True
            print("\n[REC] Recording started")
        elif c == 's' and inference_mode:
            start_recording = False
            print("\n[REC] Recording stopped")

# ======================= Inference loop =======================

def run_inference(wheelposer, config):
    """Run the main inference loop (Phase 2)"""
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
            server_for_unity.settimeout(10.0)  # 10 second timeout
            conn, addr_unity = server_for_unity.accept()
            print(f'✓ Unity connected from {addr_unity}')
        except socket.timeout:
            print('⚠ Unity connection timeout - continuing without visualization')
            conn = None
        except KeyboardInterrupt:
            print('\n⚠ Skipping Unity - continuing without visualization')
            conn = None
    
    print("\n" + "="*80)
    print("LIVE INFERENCE MODE")
    print("="*80)
    print("Controls: [r] start recording | [s] stop recording | [q] quit")
    print()
    
    # Main inference loop
    pygame.init()
    last_status_time = time.time()
    
    try:
        while inference_mode and running:
            loop_start = time.time()
            
            # Get latest IMU measurements
            ori_raw, acc_raw = get_current_imu_measurement()
            
            if ori_raw is None or acc_raw is None:
                time.sleep(0.01)
                continue
            
            # Move to device
            ori_raw = ori_raw.to(device)
            acc_raw = acc_raw.to(device)
            
            # Calibrate
            ori_raw = quaternion_to_rotation_matrix(ori_raw).view(1, 4, 3, 3)
            acc_cal = (smpl2imu.matmul(acc_raw.view(-1, 4, 3, 1)) - acc_offsets).view(1, 4, 3)
            ori_cal = smpl2imu.matmul(ori_raw).matmul(device2bone)
            imu_recording = torch.cat((acc_cal.view(-1, 12), ori_cal.view(-1, 36)), dim=1)
            
            # Normalize for model
            acc = torch.cat((acc_cal[:, :3] - acc_cal[:, 3:], acc_cal[:, 3:]), dim=1).bmm(ori_cal[:, -1]) / config.acc_scale
            ori = torch.cat((ori_cal[:, 3:].transpose(2, 3).matmul(ori_cal[:, :3]), ori_cal[:, 3:]), dim=1)
            data_nn = torch.cat((acc.view(-1, 12), ori.view(-1, 36)), dim=1)
            
            # Run inference
            pose = wheelposer.forward_online(data_nn)
            tran = torch.tensor([0, -0.4, -0.1055]).to(device)
            
            # Update FPS
            inference_fps.update(time.time())
            
            # Recording
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
                print(f'\n[REC] Saved {record_buffer.size(0)} frames at {recording_fps:.1f} FPS to {save_path / f"r{timestamp}.pt"}')
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
                # Build compact status line
                status_parts = []
                for stream in ["pocket_watch", "frame_watch", "pocket_headphone", "pocket_phone"]:
                    short_name = stream.replace("pocket_", "P-").replace("frame_", "F-").replace("phone", "Ph").replace("watch", "W").replace("headphone", "H")
                    fps_val = fps_meters[stream].get_fps(now)
                    status_parts.append(f"{short_name}:{fps_val:4.1f}")
                
                stream_status = " | ".join(status_parts)
                inf_fps = inference_fps.get_fps(now)
                rec_status = "●REC" if is_recording else "○---"
                print(f"\r[{rec_status}] {stream_status} | Inf:{inf_fps:5.1f} FPS", 
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

# ======================= Main =======================

# ======================= Main =======================

def main():
    global running, verification_mode, smpl2imu, device2bone, acc_offsets
    
    print("\n" + "="*80)
    print("WHEELPOSER LIVE INFERENCE - TWO-PHASE PIPELINE")
    print("="*80)
    print("\nDevice Configuration:")
    print("  - Left Wrist:  pocket_watch")
    print("  - Right Wrist: frame_watch")
    print("  - Head:        pocket_headphone")
    print("  - Pelvis:      pocket_phone")
    print("\nPhase 1: Stream Verification")
    print("Phase 2: Model Loading → Calibration → Inference")
    print()
    
    # ===== Setup UDP =====
    print("Setting up UDP receivers...")
    sockets = []
    for p in PORTS:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except Exception:
            pass
        sock.bind((HOST, p))
        sockets.append(sock)
    print(f"✓ Listening on ports: {PORTS}")
    
    # Start UDP receiver thread
    running = True
    verification_mode = True
    udp_thread = threading.Thread(target=udp_receiver_thread, args=(sockets,), daemon=True)
    udp_thread.start()
    
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
        # Signal threads to stop
        running = False
        time.sleep(0.5)  # Give threads time to exit select()
        # Close sockets
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
    
    try:
        input('\n[Step 1/2] Place Left Wrist sensor (pocket_watch) aligned with body frame\n'
              '           (x=Left, y=Up, z=Forward) and press Enter.')
        
        for i in range(3, 0, -1):
            print(f'\rHold steady... {i}s', end='', flush=True)
            time.sleep(1)
        
        print('\nCollecting reference orientation...')
        smpl2imu_temp, _, _ = perform_calibration(wait_seconds=3)
        print("✓ Reference frame established")
        
        input('\n[Step 2/2] Wear all 4 IMUs and stand in T-pose. Press Enter when ready.')
        for i in range(3, 0, -1):
            print(f'\rHold T-pose... {i}s', end='', flush=True)
            time.sleep(1)
        
        print('\nCollecting T-pose calibration...')
        smpl2imu, device2bone, acc_offsets = perform_calibration(wait_seconds=3)
        
        # Move to device
        smpl2imu = smpl2imu.to(device)
        device2bone = device2bone.to(device)
        acc_offsets = acc_offsets.to(device)
        
        print("✓ Calibration complete")
        print("="*80)
        
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
        
        # Wait for threads to notice running=False and exit select()
        time.sleep(0.5)
        
        # Now close sockets
        for s in sockets:
            try:
                s.close()
            except Exception:
                pass
        
        print("Shutdown complete")

if __name__ == "__main__":
    main()