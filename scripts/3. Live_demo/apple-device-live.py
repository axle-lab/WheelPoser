#  New live demo using Apple-device UDP pipeline
import sys
sys.path.append('.')

import socket
import select
import threading
import time
from datetime import datetime

import numpy as np
import torch
from pygame.time import Clock
import pygame
from pytorch_lightning import seed_everything

from src.config import Config, joint_set
from src.models.utils import get_model
from src.utils import get_checkpoints
from src.articulate.math import *
from src.models.LSTMs.Three_Stage_Global.Three_Stage_Global_WheelPoser_Wrapper import Three_Stage_Global_WheelPoser

# ===================== UDP PIPELINE CONFIG =====================
HOST = "0.0.0.0"
PORTS = [8001, 8002, 8003, 8004, 8005]
CHUNK = 8192

PLACEMENTS = ["pocket", "underseat", "back", "frame"]
SUBTYPES = ["phone", "watch", "headphone"]

KEYS = ['unix_timestamp', 'sensor_timestamp',
        'gravity_x', 'gravity_y', 'gravity_z',
        'free_accel_x', 'free_accel_y', 'free_accel_z',
        'quart_x', 'quart_y', 'quart_z', 'quart_w',
        'angular_vel_x', 'angular_vel_y', 'angular_vel_z']

# Map incoming stream keys to the 4 sensor slots expected by WheelPoser
# Adjust this mapping if your device placement differs.
STREAM_TO_SENSOR = {
    "pocket_phone": "pelvis",
    "pocket_watch": "left_arm",
    "frame_watch": "right_arm",
    "frame_headphone": "head",  # "frame_earphone" in your wording
}

# Order expected by the original live demo: leftArm, rightArm, head, pelvis
SENSOR_ORDER = ["left_arm", "right_arm", "head", "pelvis"]

# ===================== STREAM FPS MONITOR =====================

class FPSMeter:
    def __init__(self, horizon_sec=2.0):
        self.ts = []
        self.horizon = float(horizon_sec)

    def update(self, now):
        self.ts.append(now)
        horizon_start = now - self.horizon
        while self.ts and self.ts[0] < horizon_start:
            self.ts.pop(0)

    def get_fps(self, now):
        horizon_start = now - self.horizon
        while self.ts and self.ts[0] < horizon_start:
            self.ts.pop(0)
        if len(self.ts) < 2:
            return 0.0
        dur = self.ts[-1] - self.ts[0]
        return (len(self.ts) - 1) / dur if dur > 0 else 0.0

STREAM_FPS = {k: FPSMeter() for k in STREAM_TO_SENSOR.keys()}
start_model = False

def wait_for_m():
    global start_model
    while not start_model:
        c = input().strip().lower()
        if c == 'm':
            start_model = True

# ===================== UDP PARSER (from cap_imu_fps_b_folder_twoWheel.py) =====================

def parse_udp_payload(payload: bytes):
    """
    "<placement>;<type>:\n<rows...>"
    Each IMU row has 15 fields:
    unix_timestamp sensor_timestamp gravity(3) free_accel(3) quat(4) gyro(3)
    Returns list of:
      ("imu", stream_key, gravity, free_acc, quat, gyr, unix_ts, sensor_ts)
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

    if placement not in PLACEMENTS:
        return []
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
            gravity = list(map(float, fields[2:5]))
            free_acc = list(map(float, fields[5:8]))
            quat = list(map(float, fields[8:12]))   # qx, qy, qz, qw
            gyr = list(map(float, fields[12:15]))
        except ValueError:
            continue

        stream_key = f"{placement}_{dtype}"
        events.append(("imu", stream_key, gravity, free_acc, quat, gyr, unix_ts, sensor_ts))

    return events

# ===================== UDP IMU SET =====================

class UDPIMUSet:
    def __init__(self, buffer_len=26, host=HOST, ports=PORTS):
        self._quat_buffer = []
        self._acc_buffer = []
        self._buffer_len = buffer_len
        self._packet_count = 0
        self._read_thread = None
        self._is_reading = False
        self._stop = False
        self._host = host
        self._ports = ports

        # latest samples by sensor
        self._latest = {s: None for s in SENSOR_ORDER}
        self._updated = {s: False for s in SENSOR_ORDER}
        self.clock = Clock()

    def _open_sockets(self):
        sockets = []
        for p in self._ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except Exception:
                pass
            sock.bind((self._host, p))
            sockets.append(sock)
        return sockets

    def _read(self):
        sockets = self._open_sockets()
        empty = []
        try:
            while not self._stop:
                self.clock.tick(120)
                readable, _, _ = select.select(sockets, empty, empty, 0.05)
                for s in readable:
                    try:
                        data, _ = s.recvfrom(CHUNK)
                    except Exception:
                        continue
                    events = parse_udp_payload(data)
                    if not events:
                        continue
                    for ev in events:
                        if ev[0] != "imu":
                            continue
                        _, stream_key, gravity, free_acc, quat, gyr, unix_ts, sensor_ts = ev
                        if stream_key not in STREAM_TO_SENSOR:
                            continue

                        STREAM_FPS[stream_key].update(time.time())

                        sensor = STREAM_TO_SENSOR[stream_key]
                        self._latest[sensor] = (quat, free_acc)
                        self._updated[sensor] = True

                        if all(self._updated.values()):
                            full_quat = []
                            full_acc = []
                            for sname in SENSOR_ORDER:
                                q, a = self._latest[sname]
                                full_quat.append(q)
                                full_acc.append(a)

                            full_quat = np.asarray(full_quat, dtype=np.float32)  # (4,4)
                            full_acc = np.asarray(full_acc, dtype=np.float32)    # (4,3)

                            tranc = int(len(self._quat_buffer) == self._buffer_len)
                            self._quat_buffer = self._quat_buffer[tranc:] + [full_quat]
                            self._acc_buffer = self._acc_buffer[tranc:] + [full_acc]

                            self._packet_count += 1
                            for k in self._updated:
                                self._updated[k] = False
        finally:
            for s in sockets:
                try:
                    s.close()
                except Exception:
                    pass

    def start_reading(self):
        if self._read_thread is None:
            self._is_reading = True
            self._stop = False
            self._quat_buffer = []
            self._acc_buffer = []
            self._read_thread = threading.Thread(target=self._read, daemon=True)
            self._read_thread.start()
        else:
            self._quat_buffer = []
            self._acc_buffer = []
            self._is_reading = True

    def stop_reading(self):
        self._is_reading = False
        self._stop = True

    def get_current_buffer(self):
        q = torch.tensor(np.array(self._quat_buffer), dtype=torch.float32)
        a = torch.tensor(np.array(self._acc_buffer), dtype=torch.float32)
        return q, a

    def get_mean_measurement_of_n_second(self, num_seconds=3, buffer_len=120):
        save_buffer_len = self._buffer_len
        self._buffer_len = buffer_len
        self.start_reading()
        time.sleep(num_seconds)
        self.stop_reading()
        q, a = self.get_current_buffer()
        self._buffer_len = save_buffer_len
        return q.mean(dim=0), a.mean(dim=0)

# ===================== INPUT THREAD =====================

running = False
start_recording = False

def get_input():
    global running, start_recording
    while running:
        c = input().strip().lower()
        if c == 'q':
            running = False
        elif c == 'r':
            start_recording = True
        elif c == 's':
            start_recording = False

# ===================== MODEL LOADER =====================

def load_from_ckpt(config, ckpt_path, pretrained=None):
    if pretrained is None:
        model_cls = type(get_model(config))
        return model_cls.load_from_checkpoint(ckpt_path, config=config)
    model_cls = type(get_model(config, pretrained=pretrained))
    return model_cls.load_from_checkpoint(ckpt_path, config=config, pretrained_model=pretrained)

# ===================== MAIN =====================

if __name__ == '__main__':
    seed_everything(42, workers=True)

    num_past_frame = 20
    num_future_frame = 5
    physics = False
    unity_visualizer = True
    server_unity_ip = '127.0.0.1'
    server_unity_port = 8888

    # ---- IMU (UDP) streaming monitor ----
    imu_set = UDPIMUSet(buffer_len=1)
    imu_set.start_reading()
    print('UDP IMU streaming started.')
    print("Monitoring FPS. Press 'm' then Enter to load models and continue...")

    threading.Thread(target=wait_for_m, daemon=True).start()
    last_print = time.time()
    while not start_model:
        now = time.time()
        if now - last_print > 1.0:
            parts = []
            for skey in STREAM_TO_SENSOR.keys():
                parts.append(f"{skey}:{STREAM_FPS[skey].get_fps(now):5.1f}")
            print("\r" + " | ".join(parts).ljust(120), end="", flush=True)
            last_print = now
        time.sleep(0.05)

    print("\nProceeding to model loading...")

    # ---- AMASS ----
    model_names = ["IMU2Leaf_WheelPoser_AMASS", "Leaf2Full_WheelPoser_AMASS", "Full2Pose_WheelPoser_AMASS"]
    experiment_names = "TransPose_Style_500"
    leave_one_out = "am_only"

    best_ckpts = get_checkpoints(model_names, experiment_names, leave_one_out=leave_one_out)

    AMASS_IMU2Leaf_config = Config(experiment=experiment_names, model=model_names[0], project_root_dir=".", joints_set=joint_set.WheelPoser,
                                   pred_joints_set=joint_set.upper_body, normalize=True, r6d=True, loss_type="mse",
                                   use_joint_loss=False, mkdir=False, upper_body_only=True)
    AMASS_IMU2Leaf_model = load_from_ckpt(AMASS_IMU2Leaf_config, best_ckpts[model_names[0]])

    AMASS_Leaf2Full_config = Config(experiment=experiment_names, model=model_names[1], project_root_dir=".", joints_set=joint_set.WheelPoser,
                                    pred_joints_set=joint_set.upper_body, normalize=True, r6d=True, loss_type="mse",
                                    use_joint_loss=False, mkdir=False, upper_body_only=True)
    AMASS_Leaf2Full_model = load_from_ckpt(AMASS_Leaf2Full_config, best_ckpts[model_names[1]])

    AMASS_Full2Pose_config = Config(experiment=experiment_names, model=model_names[2], project_root_dir=".", joints_set=joint_set.WheelPoser,
                                    pred_joints_set=joint_set.upper_body, normalize=True, r6d=True, loss_type="mse",
                                    use_joint_loss=False, mkdir=False, upper_body_only=True)
    AMASS_Full2Pose_model = load_from_ckpt(AMASS_Full2Pose_config, best_ckpts[model_names[2]])

    # ---- WHEELPOSER ----
    model_names = ["IMU2Leaf_WheelPoser_WHEELPOSER", "Leaf2Full_WheelPoser_WHEELPOSER", "Full2Pose_WheelPoser_WHEELPOSER"]
    best_ckpts = get_checkpoints(model_names, experiment_names, leave_one_out=leave_one_out)

    WHEELPOSER_IMU2Leaf_config = Config(experiment=experiment_names, model=model_names[0], project_root_dir=".", joints_set=joint_set.WheelPoser,
                                        pred_joints_set=joint_set.upper_body, normalize=True, r6d=True, loss_type="mse",
                                        use_joint_loss=False, mkdir=False, upper_body_only=True, exp_setup='am_only', upsample_copies=7)
    WHEELPOSER_IMU2Leaf_model = load_from_ckpt(WHEELPOSER_IMU2Leaf_config, best_ckpts[model_names[0]], pretrained=AMASS_IMU2Leaf_model)

    WHEELPOSER_Leaf2Full_config = Config(experiment=experiment_names, model=model_names[1], project_root_dir=".", joints_set=joint_set.WheelPoser,
                                         pred_joints_set=joint_set.upper_body, normalize=True, r6d=True, loss_type="mse",
                                         use_joint_loss=False, mkdir=False, upper_body_only=True, exp_setup='am_only', upsample_copies=7)
    WHEELPOSER_Leaf2Full_model = load_from_ckpt(WHEELPOSER_Leaf2Full_config, best_ckpts[model_names[1]], pretrained=AMASS_Leaf2Full_model)

    WHEELPOSER_Full2Pose_config = Config(experiment=experiment_names, model=model_names[2], project_root_dir=".", joints_set=joint_set.WheelPoser,
                                         pred_joints_set=joint_set.upper_body, normalize=True, r6d=True, loss_type="mse",
                                         use_joint_loss=False, mkdir=False, upper_body_only=True, exp_setup='am_only', upsample_copies=7)
    WHEELPOSER_Full2Pose_model = load_from_ckpt(WHEELPOSER_Full2Pose_config, best_ckpts[model_names[2]], pretrained=AMASS_Full2Pose_model)

    shared_input_config = WHEELPOSER_IMU2Leaf_config
    shared_output_config = WHEELPOSER_Full2Pose_config

    wheelposer = Three_Stage_Global_WheelPoser(
        config=shared_input_config,
        imu2leaf=WHEELPOSER_IMU2Leaf_model,
        leaf2full=WHEELPOSER_Leaf2Full_model,
        full2pose=WHEELPOSER_Full2Pose_model,
        num_past_frame=num_past_frame,
        num_future_frame=num_future_frame,
        physics=physics
    ).to(shared_input_config.device)

    # ---- Calibration ----
    time.sleep(2)
    print('Check heading reset')

    oris = None
    while oris is None or oris.ndim == 0:
        q, _ = imu_set.get_current_buffer()
        if q.numel() == 0:
            print("Waiting for UDP data... start your streamer.", end="\r")
            time.sleep(0.2)
            continue
        oris = q.mean(dim=0)

    for i in range(4):
        print(quaternion_to_axis_angle(oris[i]))

    input('Put pocket_watch aligned with body reference frame (x=Left, y=Up, z=Forward) and press any key.')
    for i in range(3, 0, -1):
        print(f'\rHold the device stably. Starting in {i}...', end='')
        time.sleep(1)
    print('Keep for 3 seconds ...', end='')
    oris = imu_set.get_mean_measurement_of_n_second(num_seconds=3, buffer_len=200)[0]
    while oris.ndim == 0:
        time.sleep(0.2)
        oris = imu_set.get_mean_measurement_of_n_second(num_seconds=1, buffer_len=80)[0]
    oris = oris[0]
    smpl2imu = quaternion_to_rotation_matrix(oris).view(3, 3).t()  # global to sensor frame

    input('\tFinish.\nWear all devices correctly and press any key.')
    for i in range(3, 0, -1):
        print(f'\rStand straight in T-pose. Starting in {i}...', end='')
        time.sleep(1)
    print('\rStand straight in T-pose. Keep the pose for 3 seconds ...', end='')
    oris, accs = imu_set.get_mean_measurement_of_n_second(num_seconds=3, buffer_len=200)
    while oris.ndim == 0 or accs.ndim == 0:
        time.sleep(0.2)
        oris, accs = imu_set.get_mean_measurement_of_n_second(num_seconds=1, buffer_len=80)

    oris = quaternion_to_rotation_matrix(oris)  # sensor to global
    device2bone = smpl2imu.matmul(oris).transpose(1, 2).matmul(torch.eye(3))
    acc_offsets = smpl2imu.matmul(accs.unsqueeze(-1))

    smpl2imu = smpl2imu.to(shared_input_config.device)
    device2bone = device2bone.to(shared_input_config.device)
    acc_offsets = acc_offsets.to(shared_input_config.device)

    print('\tFinish.\nStart estimating poses.')
    running = True
    old_timestamp = 0
    is_recording = False
    record_buffer = None

    # Unity
    if unity_visualizer:
        server_for_unity = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_for_unity.bind((server_unity_ip, server_unity_port))
        server_for_unity.listen(5)
        print('Server start. Waiting for Unity to connect.')
        conn, addr_unity = server_for_unity.accept()
        print('Got connection from', addr_unity)

    # input thread
    get_input_thread = threading.Thread(target=get_input, daemon=True)
    get_input_thread.start()

    wheelposer.eval()
    pygame.init()

    while running:
        start_time = time.time()
        ori_raw, acc_raw = imu_set.get_current_buffer()
        ori_raw = ori_raw.to(shared_input_config.device)
        acc_raw = acc_raw.to(shared_input_config.device)

        if (ori_raw.size(0) != 0) and (acc_raw.size(0) != 0) and (old_timestamp != imu_set._packet_count):
            old_timestamp = imu_set._packet_count

            ori_raw = quaternion_to_rotation_matrix(ori_raw).view(1, 4, 3, 3)
            acc_cal = (smpl2imu.matmul(acc_raw.view(-1, 4, 3, 1)) - acc_offsets).view(1, 4, 3)
            ori_cal = smpl2imu.matmul(ori_raw).matmul(device2bone)
            imu_recording = torch.cat((acc_cal.view(-1, 12), ori_cal.view(-1, 36)), dim=1)

            acc = torch.cat((acc_cal[:, :3] - acc_cal[:, 3:], acc_cal[:, 3:]), dim=1).bmm(ori_cal[:, -1]) / WHEELPOSER_Full2Pose_config.acc_scale
            ori = torch.cat((ori_cal[:, 3:].transpose(2, 3).matmul(ori_cal[:, :3]), ori_cal[:, 3:]), dim=1)
            data_nn = torch.cat((acc.view(-1, 12), ori.view(-1, 36)), dim=1)

            pose = wheelposer.forward_online(data_nn)
            tran = torch.tensor([0, -0.4, -0.1055]).to(shared_input_config.device)

            # recording (same behavior as old script)
            if not is_recording and start_recording:
                record_buffer = imu_recording.view(1, -1)
                is_recording = True
                rec_start = time.time()
            elif is_recording and start_recording:
                record_buffer = torch.cat([record_buffer, imu_recording.view(1, -1)], dim=0)
            elif is_recording and not start_recording:
                rec_end = time.time()
                torch.save(record_buffer, 'src/data/imu_recordings/r' + datetime.now().strftime('%T').replace(':', '-') + '.pt')
                is_recording = False
                fps = record_buffer.size(0) / max(1e-6, (rec_end - rec_start))
                print('Recording FPS is:', fps)

            s = ','.join(['%g' % v for v in pose]) + '#' + ','.join(['%g' % v for v in tran]) + '$'
            if unity_visualizer:
                conn.send(s.encode('utf8'))
            print(f"FPS: {1/(time.time()-start_time)}")