#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified BLE + UDP collector with real-time FPS printing (short labels, decay to 0 on inactivity).
Modified to support dual BLE devices: LeftWheel and RightWheel simultaneously.

- UDP message format aligned with your original script:
    "<placement>;<type>:\\n<rows...>"
    "<placement>;phone_watch_distance:<unix_ts> <distance>"
  placement ∈ {pocket, underseat, back, frame}
  type ∈ {phone, watch}
  Each IMU row has 12 fields (space-separated):
    unix_timestamp sensor_timestamp ax ay az qx qy qz qw gx gy gz

- Keyboard controls:
    r : start recording
    s : stop recording and flush files
    b : check BLE connection and reconnect if needed
    q : quit
"""

from pathlib import Path
import socket
import select
import numpy as np
import os
import time
import threading
from datetime import datetime
import argparse
import asyncio
from bleak import BleakScanner, BleakClient
from collections import deque
import struct

# ======================= Configuration =======================

HOST = "0.0.0.0"
PORTS = [8001, 8002, 8003, 8004, 8005]
CHUNK = 8192
BUFFER_SIZE = 300
STATUS_INTERVAL = 1.0  # seconds between status prints

# Devices & streams
PLACEMENTS = ["pocket", "underseat", "back", "frame"]
SUBTYPES   = ["phone", "watch", "headphone"]  # watch for pocket & frame

# Row keys (original order)
KEYS = ['unix_timestamp', 'sensor_timestamp',
        'gravity_x', 'gravity_y', 'gravity_z',
        'free_accel_x', 'free_accel_y', 'free_accel_z',
        'quart_x', 'quart_y', 'quart_z', 'quart_w',
        'angular_vel_x', 'angular_vel_y', 'angular_vel_z']

# BLE notify UUID
BLE_NOTIFY_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

# ======================= Global state =======================

running = False
start_recording = False
record_session_start = 0.0
participant_dir = None  # Will store the participant directory

# Recording per-stream
is_recording = {}            # stream_key -> bool
record_buffers = {}          # stream_key -> list[str]
record_start_times = {}      # stream_key -> float
record_counts = {}           # stream_key -> int

# Distance data (per side)
distance_records = {"pocket": [], "frame": []}

# BLE data - separate records for each device
ble_records = {
    "LeftWheel": [],         # list of (unix_ts, ch0, ch1, ch2, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
    "RightWheel": []
}

# Time alignment per stream
reference_times = {}         # stream_key -> [unix0, sensor0]
last_timestamps = {}         # stream_key -> float
sampling_rates = {}          # stream_key -> float

# Optional rolling buffers
raw_gravity_buffer = {}      # stream_key -> (BUFFER_SIZE,3)
raw_acc_buffer = {}          # stream_key -> (BUFFER_SIZE,3)
raw_ori_buffer = {}          # stream_key -> (BUFFER_SIZE,4)
raw_angular_vel_buffer = {}  # stream_key -> (BUFFER_SIZE,3)

# BLE control - separate threads for each device
ble_threads = {}
ble_reconnect_requested = {"LeftWheel": False, "RightWheel": False}

# ======================= FPS meters with decay =======================

class FPSMeter:
    """
    Maintains a sliding window of timestamps and reports FPS.
    Timestamps older than horizon_sec are dropped during get_fps(now),
    so when the stream stops, FPS decays to 0 after ~horizon_sec.
    """
    def __init__(self, horizon_sec=2.0, maxlen=200):
        self.ts = deque(maxlen=maxlen)
        self.horizon = float(horizon_sec)

    def update(self, now: float):
        self.ts.append(now)

    def get_fps(self, now: float) -> float:
        # Drop timestamps outside the horizon window
        horizon_start = now - self.horizon
        while self.ts and self.ts[0] < horizon_start:
            self.ts.popleft()
        if len(self.ts) < 2:
            return 0.0
        dur = self.ts[-1] - self.ts[0]
        return (len(self.ts) - 1) / dur if dur > 0 else 0.0

# Per-stream FPS (stream_key = "<placement>_<type>")
fps_meters = {
    "pocket_phone": FPSMeter(),
    "pocket_watch": FPSMeter(),
    "underseat_phone": FPSMeter(),
    "back_phone": FPSMeter(),
    "frame_phone": FPSMeter(),
    "frame_watch": FPSMeter(),
    "underseat_headphone": FPSMeter(),
    # "pocket_headphone": FPSMeter(),

}
# Distance FPS
distance_fps_meters = {
    "pocket": FPSMeter(),
    "frame": FPSMeter(),
}
# BLE FPS - separate for each device
ble_fps_meters = {
    "LeftWheel": FPSMeter(),
    "RightWheel": FPSMeter()
}

last_status_print_time = time.time()

# ======================= Helpers =======================

def ensure_stream_state(stream_key):
    if stream_key not in is_recording:
        is_recording[stream_key] = False
        record_buffers[stream_key] = []
        record_counts[stream_key] = 0
    if stream_key not in reference_times:
        reference_times[stream_key] = None
    if stream_key not in last_timestamps:
        last_timestamps[stream_key] = None
    if stream_key not in sampling_rates:
        sampling_rates[stream_key] = None
    if stream_key not in raw_gravity_buffer:
        raw_gravity_buffer[stream_key] = np.zeros((BUFFER_SIZE, 3))
    if stream_key not in raw_acc_buffer:
        raw_acc_buffer[stream_key] = np.zeros((BUFFER_SIZE, 3))
    if stream_key not in raw_ori_buffer:
        raw_ori_buffer[stream_key] = np.array([[0, 0, 0, 1]] * BUFFER_SIZE)
    if stream_key not in raw_angular_vel_buffer:
        raw_angular_vel_buffer[stream_key] = np.zeros((BUFFER_SIZE, 3))

def create_recording_directory(ts_label=None):
    global participant_dir
    if ts_label is None:
        # When called without parameter
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    else:
        # When called WITH shared timestamp (e.g., "14-30-45")
        today = datetime.now().strftime('%Y-%m-%d')  # ← Gets year-month-day
        timestamp = f"{today}_{ts_label}"            # ← Combines them
    
    recording_dir = participant_dir / f"recording_{timestamp}"
    recording_dir.mkdir(exist_ok=True, parents=True)
    return recording_dir

# ======================= BLE client =======================

class BLEGroundTruth(threading.Thread):
    """
    Connects to BLE device by name and collects notifications.
    Each notification expected as 9 * int32 (little-endian): 3 capacitive + 6 IMU.
    """
    def __init__(self, device_name):
        super().__init__(daemon=True)
        self.device_name = device_name
        self._stop = threading.Event()
        self.connected = False
        self.client = None

    async def _connect_and_listen(self):
        while not self._stop.is_set():
            if ble_reconnect_requested[self.device_name]:
                ble_reconnect_requested[self.device_name] = False
                await self._attempt_connection()
            else:
                # Just maintain existing connection if connected, otherwise wait
                if self.connected and self.client and self.client.is_connected:
                    # Connection is alive, just wait
                    await asyncio.sleep(0.1)
                else:
                    # Not connected, but don't auto-reconnect - wait for manual trigger
                    self.connected = False
                    await asyncio.sleep(0.5)

    async def _attempt_connection(self):
        try:
            print(f"\n[BLE] Scanning for '{self.device_name}'...")
            dev = await BleakScanner.find_device_by_name(self.device_name)
            if dev is None:
                print(f"[BLE][WARN] Device '{self.device_name}' not found.")
                self.connected = False
                return

            def _dc_cb(_client):
                print(f"\n[BLE] {self.device_name} disconnected.")
                self.connected = False

            self.client = BleakClient(dev.address, _dc_cb)
            await self.client.connect()
            self.connected = True
            print(f"[BLE] Connected to '{self.device_name}'. Subscribing notifications...")
            
            await self.client.start_notify(BLE_NOTIFY_UUID, self._on_notify)
            
            # Keep connection alive until disconnection or stop requested
            while self.connected and not self._stop.is_set() and not ble_reconnect_requested[self.device_name]:
                # Check if client is still actually connected
                if not self.client.is_connected:
                    self.connected = False
                    break
                await asyncio.sleep(0.1)
                
            if self.client and self.client.is_connected:
                try:
                    await self.client.stop_notify(BLE_NOTIFY_UUID)
                    await self.client.disconnect()
                except Exception:
                    pass
                    
        except Exception as e:
            print(f"\n[BLE][ERR] {self.device_name}: {e}")
            self.connected = False

    def _on_notify(self, _char, data: bytearray):
        # Expect 10 * 4 bytes = 40 bytes total
        expected_len = 10 * 4
        if len(data) != expected_len:
            print(f"\n[BLE][WARN] {self.device_name}: Expected {expected_len} bytes, got {len(data)}")
            return
        
        # Parse capacitive sensor data (first 3 uint32 values)
        cap_vals = []
        for i in range(3):
            cap_vals.append(int.from_bytes(data[4*i:4*i+4], "little", signed=False))
        
        # Parse IMU data (next 6 uint32 values, convert back to float)
        imu_vals = []
        for i in range(3, 9):
            # Convert uint32 back to float32
            uint32_val = int.from_bytes(data[4*i:4*i+4], "little", signed=False)
            float_val = struct.unpack('<f', struct.pack('<I', uint32_val))[0]
            imu_vals.append(float_val)

        arduino_timestamp = int.from_bytes(data[36:40], "little", signed=False)
        
        # Split IMU data into accelerometer and gyroscope
        accel_x, accel_y, accel_z = imu_vals[0], imu_vals[1], imu_vals[2]
        gyro_x, gyro_y, gyro_z = imu_vals[3], imu_vals[4], imu_vals[5]

        now = time.time()
        ble_fps_meters[self.device_name].update(now)
        
        if start_recording:
            # Append all data as a single tuple: (timestamp, cap0, cap1, cap2, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
            ble_records[self.device_name].append((now, cap_vals[0], cap_vals[1], cap_vals[2], 
                                                 accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, arduino_timestamp))

    def run(self):
        try:
            asyncio.run(self._connect_and_listen())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._connect_and_listen())

    def stop(self):
        self._stop.set()
        self.connected = False

    def is_connected(self):
        return self.connected

    def request_reconnect(self):
        ble_reconnect_requested[self.device_name] = True

# ======================= UDP parsing (original-aligned) =======================

def parse_udp_payload(payload: bytes):
    """
    Original-aligned parser:
      "<placement>;<type>:\\n<rows...>"
      "<placement>;phone_watch_distance:<unix_ts> <distance>"
    Returns list of events:
      ("distance", side, ts, dist)
      ("imu", stream_key, save_line, acc, quat, gyr, unix_ts, sensor_ts)
    """
    try:
        msg = payload.decode("utf-8").strip()
    except Exception:
        return []

    if ';' not in msg or ':' not in msg:
        print(msg)
        return []

    try:
        placement, rhs = msg.split(";", 1)
        placement = placement.strip()
        dtype, body = rhs.split(":", 1)
        dtype = dtype.strip()
        body = body.strip()
    except ValueError:
        return []

    if placement not in PLACEMENTS:
        return []

    # Distance message
    if dtype == "phone_watch_distance":
        parts = body.split()
        if len(parts) == 2:
            try:
                d_time = float(parts[0])
                d_val = float(parts[1])
                if placement in ("pocket", "frame"):
                    return [("distance", placement, d_time, d_val)]
            except ValueError:
                pass
        return []

    # IMU batch (type = phone | watch)
    if dtype not in SUBTYPES:
        return []

    events = []
    for line in body.splitlines():
        fields = line.strip().split()
        if len(fields) != len(KEYS):
            continue
        try:
            unix_ts = float(fields[0])
            sensor_ts = float(fields[1])
            gravity = list(map(float, fields[2:5]))      # gravity_x, gravity_y, gravity_z
            free_acc = list(map(float, fields[5:8]))     # free_accel_x, free_accel_y, free_accel_z
            quat = list(map(float, fields[8:12]))        # quart_x, quart_y, quart_z, quart_w
            gyr = list(map(float, fields[12:15]))        # angular_vel_x, angular_vel_y, angular_vel_z
        except ValueError:
            continue

        stream_key = f"{placement}_{dtype}"
        save_line = ",".join([fields[0], fields[1]] + [str(x) for x in gravity + free_acc + quat + gyr])
        events.append(("imu", stream_key, save_line, gravity, free_acc, quat, gyr, unix_ts, sensor_ts))

    return events

# ======================= Recording helpers =======================

def start_stream_recording(stream_key):
    record_buffers[stream_key] = []
    is_recording[stream_key] = True
    record_start_times[stream_key] = time.time()
    record_counts[stream_key] = 0

def stop_stream_recording_and_flush(stream_key, ts_label = None):
    if not is_recording.get(stream_key, False):
        return
    end_time = time.time()
    ts_label = ts_label or datetime.now().strftime('%H-%M-%S')
    rows = record_buffers.get(stream_key, [])
    
    # Create recording directory only when saving
    out_dir = create_recording_directory(ts_label)
    
    fname = f"{stream_key}-recording-{ts_label}.csv"
    with open(out_dir / fname, "w") as f:
        f.write(",".join(KEYS) + "\n")
        f.write("\n".join(rows))

    # ---------- NPY ----------
    # Convert to float array (N, len(KEYS))
    arr = np.array([list(map(float, r.split(","))) for r in rows], dtype=np.float64)
    fname_npy = f"{stream_key}-recording-{ts_label}.npy"
    np.save(out_dir / fname_npy, arr)

    dur = max(1e-6, end_time - record_start_times.get(stream_key, end_time))
    fps = record_counts.get(stream_key, 0) / dur
    print(f"\n[WRITE] {fname} | {fps:.2f} FPS | {record_counts.get(stream_key, 0)} rows")
    is_recording[stream_key] = False

def stop_distance_and_flush(side_key, ts_label = None):
    end_time = time.time()
    ts_label = ts_label or datetime.now().strftime('%H-%M-%S')
    rows = distance_records.get(side_key, [])
    if not rows:
        return
    
    # Create recording directory only when saving
    out_dir = create_recording_directory(ts_label)
    
    fname = f"{side_key}-phone_watch_distance-{ts_label}.csv"
    with open(out_dir / fname, "w") as f:
        f.write("timestamp,distance\n")
        for t, d in rows:
            f.write(f"{t},{d}\n")

    # ---------- NPY ----------
    arr = np.array(rows, dtype=np.float64)  # shape (N, 2)
    fname_npy = f"{side_key}-phone_watch_distance-{ts_label}.npy"
    np.save(out_dir / fname_npy, arr)

    dur = max(1e-6, end_time - record_session_start)
    fps = len(rows) / dur
    print(f"\n[WRITE] {fname} | {fps:.2f} FPS | {len(rows)} rows")
    rows.clear()

def stop_ble_and_flush(ts_label = None):
    """Flush BLE data for both devices"""
    end_time = time.time()
    ts_label = ts_label or datetime.now().strftime('%H-%M-%S')

    for device_name in ["LeftWheel", "RightWheel"]:
        records = ble_records[device_name]
        if not records:
            continue
        
        # Create recording directory only when saving
        out_dir = create_recording_directory(ts_label)
        
        fname = f"ble_sensors_{device_name.lower()}-recording-{ts_label}.csv"
        with open(out_dir / fname, "w") as f:
            # Updated header to include all sensor data
            f.write("unix_timestamp,cap_ch0,cap_ch1,cap_ch2,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,arduino_timestamp_ms\n")
            for row in records:
                # Write all 10 values: timestamp + 3 capacitive + 6 IMU
                f.write("{:.6f},{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{}\n".format(
                    row[0],    # timestamp
                    row[1],    # cap_ch0
                    row[2],    # cap_ch1  
                    row[3],    # cap_ch2
                    row[4],    # accel_x
                    row[5],    # accel_y
                    row[6],    # accel_z
                    row[7],    # gyro_x
                    row[8],    # gyro_y
                    row[9],     # gyro_z
                    row[10]    # arduino_timestamp
                ))

        # ---------- NPY ----------
        arr = np.array(records, dtype=np.float64)  # shape (N, 11)
        fname_npy = f"ble_sensors_{device_name.lower()}-recording-{ts_label}.npy"
        np.save(out_dir / fname_npy, arr)

        dur = max(1e-6, end_time - record_session_start)
        fps = len(records) / dur
        print(f"\n[WRITE] {fname} | {fps:.2f} FPS | {len(records)} rows | {device_name} Capacitive + IMU data")
        records.clear()

# ======================= BLE connection management =======================

def check_ble_connection():
    """Check BLE connection status and attempt reconnection if needed."""
    for device_name in ["LeftWheel", "RightWheel"]:
        ble_thread = ble_threads.get(device_name)
        if ble_thread is None:
            print(f"\n[BLE] {device_name} thread not initialized.")
            continue
        
        if ble_thread.is_connected():
            print(f"\n[BLE] Connected to '{device_name}' ✓")
        else:
            print(f"\n[BLE] Not connected to '{device_name}'. Attempting reconnection...")
            ble_thread.request_reconnect()

# ======================= Keyboard control =======================

def input_thread():
    global running, start_recording, record_session_start
    while running:
        try:
            c = input().strip().lower()
        except EOFError:
            break
        if c == 'q':
            running = False
        elif c == 'r':
            if not start_recording:
                start_recording = True
                record_session_start = time.time()
                print(f"\n[REC] Recording started")
        elif c == 's':
            if start_recording:
                start_recording = False
                print("\n[REC] Stopping & flushing...")
        elif c == 'b':
            check_ble_connection()

# ======================= Main =======================

def main():
    global running, last_status_print_time, record_session_start, ble_threads, participant_dir

    parser = argparse.ArgumentParser(description="Unified BLE + UDP collector (dual BLE devices: LeftWheel + RightWheel)")
    args = parser.parse_args()

    # Prepare output directory structure
    save_root = Path("study_data")
    save_root.mkdir(exist_ok=True, parents=True)
    n_participants = len([x for x in save_root.iterdir() if x.is_dir()])
    pid = input("Enter a participant name: ").strip()
    participant_dir = save_root / f"{n_participants + 1}_{pid}"
    participant_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"[SETUP] Participant directory: {participant_dir}")

    # Start BLE threads for both devices (but don't auto-connect)
    for device_name in ["LeftWheel", "RightWheel"]:
        ble_thread = BLEGroundTruth(device_name=device_name)
        ble_thread.start()
        ble_threads[device_name] = ble_thread
        print(f"[BLE] {device_name} thread started.")
    
    print(f"[BLE] Press 'b' to connect to both devices")

    # Bind UDP sockets
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
        print(f"[UDP] Listening on {HOST}:{p}")

    # Keyboard thread
    running = True
    record_session_start = time.time()
    threading.Thread(target=input_thread, daemon=True).start()
    
    print("\n[CONTROLS] r=start recording | s=stop recording | b=check BLE | q=quit")

    empty = []

    try:
        while running:
            readable, _, _ = select.select(sockets, empty, empty, 0.1)

            for s in readable:
                try:
                    data, _ = s.recvfrom(CHUNK)
                except Exception:
                    continue

                events = parse_udp_payload(data)
                if not events:
                    continue

                for ev in events:
                    if ev[0] == "distance":
                        _, side, d_time, d_val = ev
                        # distance FPS
                        distance_fps_meters[side].update(time.time())
                        if start_recording:
                            distance_records[side].append((d_time, d_val))
                        continue

                    # IMU event
                    _, stream_key, save_line, gravity, free_acc, quat, gyr, unix_ts, sensor_ts = ev
                    ensure_stream_state(stream_key)

                    # rolling buffers (optional)
                    raw_gravity_buffer[stream_key] = np.concatenate(
                        [raw_gravity_buffer[stream_key][1:], np.array(gravity).reshape(1, 3)]
                    )
                    raw_acc_buffer[stream_key] = np.concatenate(
                        [raw_acc_buffer[stream_key][1:], np.array(free_acc).reshape(1, 3)]
                    )
                    raw_ori_buffer[stream_key] = np.concatenate(
                        [raw_ori_buffer[stream_key][1:], np.array(quat).reshape(1, 4)]
                    )
                    raw_angular_vel_buffer[stream_key] = np.concatenate(
                        [raw_angular_vel_buffer[stream_key][1:], np.array(gyr).reshape(1, 3)]
                    )

                    # time alignment (like your original)
                    if reference_times[stream_key] is None:
                        reference_times[stream_key] = [unix_ts, sensor_ts]
                    curr_ts = reference_times[stream_key][0] + (sensor_ts - reference_times[stream_key][1])
                    if last_timestamps[stream_key] is not None:
                        dt = curr_ts - last_timestamps[stream_key]
                        if dt > 0:
                            sampling_rates[stream_key] = 1.0 / dt
                    last_timestamps[stream_key] = curr_ts

                    # FPS update
                    if stream_key in fps_meters:
                        fps_meters[stream_key].update(time.time())

                    # recording logic
                    if start_recording:
                        if not is_recording[stream_key]:
                            start_stream_recording(stream_key)
                        record_buffers[stream_key].append(save_line)
                        record_counts[stream_key] += 1

            # When not recording, flush once for any active streams
            if not start_recording:
                has_data_to_flush = (
                    any(is_recording.get(k, False) for k in is_recording.keys()) or
                    distance_records.get("pocket") or
                    distance_records.get("frame") or
                    ble_records.get("LeftWheel") or
                    ble_records.get("RightWheel")
                )
                
                if has_data_to_flush:
                    # Generate ONE shared timestamp for all files in this flush
                    shared_ts_label = datetime.now().strftime('%H-%M-%S')
                    
                    for k in list(is_recording.keys()):
                        if is_recording[k]:
                            stop_stream_recording_and_flush(k, ts_label=shared_ts_label)
                    stop_distance_and_flush("pocket", ts_label=shared_ts_label)
                    stop_distance_and_flush("frame", ts_label=shared_ts_label)
                    stop_ble_and_flush(ts_label=shared_ts_label)
                    
                    print(f"\n[REC] All files saved with timestamp: {shared_ts_label}")

            # ===== Real-time FPS print (short labels, decay-aware) =====
            now = time.time()
            if now - last_status_print_time > STATUS_INTERVAL:
                # Add BLE connection status for both devices
                left_status = "CONN" if ble_threads.get("LeftWheel") and ble_threads["LeftWheel"].is_connected() else "DISC"
                right_status = "CONN" if ble_threads.get("RightWheel") and ble_threads["RightWheel"].is_connected() else "DISC"
                
                # Short labels: Pk-P, Pk-W, Us-P, Bk-P, Fr-P, Fr-W, DistPk, DistFr, BLE-L, BLE-R
                parts = [
                    f"Rec:{'ON' if start_recording else 'OFF'}",
                    f"BLE-L:{left_status}",
                    f"BLE-R:{right_status}",
                    f"Pk-P:{fps_meters['pocket_phone'].get_fps(now):5.1f}",
                    f"Pk-W:{fps_meters['pocket_watch'].get_fps(now):5.1f}",
                    f"Fr-P:{fps_meters['frame_phone'].get_fps(now):5.1f}",
                    f"Fr-W:{fps_meters['frame_watch'].get_fps(now):5.1f}",
                    f"Us-P:{fps_meters['underseat_phone'].get_fps(now):5.1f}",
                    f"Us-H:{fps_meters['underseat_headphone'].get_fps(now):5.1f}",
                    f"Bk-P:{fps_meters['back_phone'].get_fps(now):5.1f}",
                    # f"DistPk:{distance_fps_meters['pocket'].get_fps(now):4.1f}",
                    # f"DistFr:{distance_fps_meters['frame'].get_fps(now):4.1f}",
                    f"BLE-L:{ble_fps_meters['LeftWheel'].get_fps(now):4.1f}",
                    f"BLE-R:{ble_fps_meters['RightWheel'].get_fps(now):4.1f}",
                ]
                print("\r" + " | ".join(parts).ljust(100), end="", flush=True)
                last_status_print_time = now

    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted.")
    finally:
        running = False
        if start_recording:
            shared_ts_label = datetime.now().strftime('%H-%M-%S')
            for k in list(is_recording.keys()):
                if is_recording[k]:
                    stop_stream_recording_and_flush(k, shared_ts_label)
            stop_distance_and_flush("pocket", shared_ts_label)
            stop_distance_and_flush("frame", shared_ts_label)
            stop_ble_and_flush(shared_ts_label)
        for s in sockets:
            try:
                s.close()
            except Exception:
                pass
        for ble_thread in ble_threads.values():
            try:
                ble_thread.stop()
            except Exception:
                pass
        print("\n[MAIN] Exit.")

if __name__ == "__main__":
    main()