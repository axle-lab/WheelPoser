
#  Copyright (c) 2003-2023 Movella Technologies B.V. or subsidiaries worldwide.
#  All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without modification,
#  are permitted provided that the following conditions are met:
#  
#  1.	Redistributions of source code must retain the above copyright notice,
#  	this list of conditions and the following disclaimer.
#  
#  2.	Redistributions in binary form must reproduce the above copyright notice,
#  	this list of conditions and the following disclaimer in the documentation
#  	and/or other materials provided with the distribution.
#  
#  3.	Neither the names of the copyright holders nor the names of their contributors
#  	may be used to endorse or promote products derived from this software without
#  	specific prior written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
#  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
#  THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
#  OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR
#  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  

# Requires installation of the correct Movella DOT PC SDK wheel through pip
# For example, for Python 3.9 on Windows 64 bit run the following command
# pip install movelladot_pc_sdk-202x.x.x-cp39-none-win_amd64.whl

import sys
sys.path.append('.')

from src.dot_sdk.xdpchandler import *

from concurrent.futures import thread
from http import server
import socket
import threading
from datetime import datetime
import torch
import numpy as np
import time
# from net import TransPoseNet
import matplotlib.pyplot as plt
import argparse
import pickle as pkl
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from src.config import Config, joint_set
from src.evaluator import ReducedPoseEvaluator, WheelPoserEvaluator
from src.models.utils import get_model
from src.data.utils import get_datamodule, get_dataset
from src.articulate.math import *
from src.utils import *
from pygame.time import Clock
import pygame
from src.models.LSTMs.Three_Stage_Global.Three_Stage_Global_WheelPoser_Wrapper import Three_Stage_Global_WheelPoser

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# inertial_poser = TransPoseNet(num_past_frame=20,num_future_frame=5).to(device)



running = False
start_recording = False
unity_visualizer = True

server_unity_ip = '127.0.0.1'
server_unity_port = 8888

xdpcHandler = XdpcHandler()
imu_count = 4

start_time = None
end_time = None

#D4:22:CD:00:50:D2 Pelvis
#D4:22:CD:00:50:D1 LeftArm
#D4:22:CD:00:50:DE LeftLeg
#D4:22:CD:00:53:AA Head



class IMUSet:
    def __init__(self, buffer_len = 26):

        self._quat_buffer = []
        self._acc_buffer = []
        self._read_thread = None
        self._is_reading = False
        self._head_buffer = []
        self._leftArm_buffer = []
        self._rightArm_buffer = []
        self._pelvis_buffer = []
        self._packet_count = 0
        self._buffer_len = buffer_len
        self.clock = Clock()


    
    def establish_connection(self):
        if not xdpcHandler.initialize():
            xdpcHandler.cleanup()
            exit(-1)

        xdpcHandler.scanForDots(num_expected=4)
        if len(xdpcHandler.detectedDots()) == 0:
            print("No Movella DOT device(s) found. Aborting.")
            xdpcHandler.cleanup()
            exit(-1)

        xdpcHandler.connectDots()

        if len(xdpcHandler.connectedDots()) == 0:
            print("Could not connect to any Movella DOT device(s). Aborting.")
            xdpcHandler.cleanup()
            exit(-1)

        for device in xdpcHandler.connectedDots():
            # Make sure all connected devices have the same filter profile and output rate
            if device.setOnboardFilterProfile("General"):
                print("Successfully set profile to General")
            else:
                print("Setting filter profile failed!")

            if device.setOutputRate(60):
                print("Successfully set output rate to 60 Hz")
            else:
                print("Setting output rate failed!")

        manager = xdpcHandler.manager()
        deviceList = xdpcHandler.connectedDots()
        # print(f"\nStarting sync for connected devices... Root node: {deviceList[-1].bluetoothAddress()}")
        print(f"\nStarting sync for connected devices... Root node: D4:22:CD:00:53:AA")
        print("This takes at least 14 seconds")
        # if not manager.startSync(deviceList[-1].bluetoothAddress()):
        if not manager.startSync("D4:22:CD:00:53:AA"):
            print(f"Could not start sync. Reason: {manager.lastResultText()}")
            if manager.lastResult() != movelladot_pc_sdk.XRV_SYNC_COULD_NOT_START:
                print("Sync could not be started. Aborting.")
                xdpcHandler.cleanup()
                exit(-1)

            # If (some) devices are already in sync mode.Disable sync on all devices first.
            manager.stopSync()
            print(f"Retrying start sync after stopping sync")
            if not manager.startSync(deviceList[-1].bluetoothAddress()):
                print(f"Could not start sync. Reason: {manager.lastResultText()}. Aborting.")
                xdpcHandler.cleanup()
                exit(-1)

        # Start live data output. Make sure root node is last to go to measurement.
        print("Putting devices into measurement mode.")
        for device in xdpcHandler.connectedDots():
            if not device.startMeasurement(movelladot_pc_sdk.XsPayloadMode_CompleteQuaternion):
                print(f"Could not put device into measurement mode. Reason: {device.lastResultText()}")
                continue
        
    def _read(self):
        while True:
            self.clock.tick(120)
            if xdpcHandler.packetsAvailable():
                # self.clock.tick()
                # batch = xdpcHandler.getNextBatch()
                for device in xdpcHandler.connectedDots():
                    packet = xdpcHandler.getNextPacket(device.portInfo().bluetoothAddress())
                    if packet.containsOrientation() and packet.containsFreeAcceleration():
                        quat = packet.orientationQuaternion()
                        acc = packet.freeAcceleration()
                        quat = np.append(quat,acc)
                        if device.deviceTagName() == 'Head':
                            self._head_buffer.append(quat)
                        elif device.deviceTagName() == 'Pelvis':
                            self._pelvis_buffer.append(quat)
                        elif device.deviceTagName() == 'LeftArm':
                            self._leftArm_buffer.append(quat)
                        # elif device.deviceTagName() == 'RightArm':
                        elif device.deviceTagName() == 'LeftLeg':
                            self._rightArm_buffer.append(quat)
                #if len(self._head_buffer)>=1 and len(self._pelvis_buffer)>=1 and len(self._leftArm_buffer)>=1 and len(self._rightArm_buffer)>=1 and len(self._leftLeg_buffer)>=1:
                self._packet_count+=1
                if self._is_reading:
                    full_measurement = np.concatenate((self._leftArm_buffer[0], self._rightArm_buffer[0],self._head_buffer[0],self._pelvis_buffer[0])).reshape((4,7))
                    tranc = int(len(self._quat_buffer) == self._buffer_len)
                    self._quat_buffer = self._quat_buffer[tranc:] + [full_measurement[:,:4].astype(float)]
                    self._acc_buffer = self._acc_buffer[tranc:] + [full_measurement[:,4:7].astype(float)]
                self._head_buffer.pop(0)
                self._leftArm_buffer.pop(0)
                self._rightArm_buffer.pop(0)
                self._pelvis_buffer.pop(0)

    
    
    def start_reading(self):
        if self._read_thread is None:
            self._is_reading = True
            self._quat_buffer = []
            self._acc_buffer = []
            self._read_thread = threading.Thread(target=self._read)
            self._read_thread.setDaemon(True)
            self._read_thread.start()
        else:
            self._quat_buffer = []
            self._acc_buffer = []
            # print('Failed to start reading thread: reading thread is already start.')
            print('cleared the buffer')
            self._is_reading = True



    
    def stop_reading(self):
        if self._read_thread is not None:
            self._is_reading = False



    def get_current_buffer(self):
        q = torch.tensor(np.array(self._quat_buffer), dtype=torch.float32)
        a = torch.tensor(np.array(self._acc_buffer), dtype=torch.float32)
        return q, a 

    def get_mean_measurement_of_n_second(self, num_seconds=3, buffer_len=120):
        """
        Start reading for `num_seconds` seconds and then close the connection. The average of the last
        `buffer_len` frames of the measured quaternions and accelerations are returned.
        Note that this function is blocking.

        :param num_seconds: How many seconds to read.
        :param buffer_len: Buffer length. Must be smaller than 60 * `num_seconds`.
        :return: The mean quaternion and acceleration torch.Tensor in shape [6, 4] and [6, 3] respectively.
        """
        save_buffer_len = self._buffer_len
        self._buffer_len = buffer_len


        self.start_reading()
        time.sleep(num_seconds)
        self.stop_reading()
        q, a = self.get_current_buffer()
        print(q.shape)
        self._buffer_len = save_buffer_len
        return q.mean(dim=0), a.mean(dim=0)

def get_input():
    global running, start_recording
    while running:
        c = input()
        if c == 'q':
            running = False
        elif c == 'r':
            start_recording = True
        elif c == 's':
            start_recording = False



if __name__ == '__main__':
    #Setup the Model
    seed_everything(42, workers=True)

    num_past_frame = 20
    num_future_frame = 5
    physics = False


    model_names = ["IMU2Leaf_WheelPoser_AMASS", "Leaf2Full_WheelPoser_AMASS", "Full2Pose_WheelPoser_AMASS"]
    experiment_names = "TransPose_Style_500"
    leave_one_out = 'am_only'

    # %%
    best_ckpts = get_checkpoints(model_names, experiment_names, leave_one_out=leave_one_out)
    print(best_ckpts)

    # %%

    AMASS_IMU2Leaf_config = Config(experiment=experiment_names, model=model_names[0], project_root_dir=".", joints_set=joint_set.WheelPoser, pred_joints_set=joint_set.upper_body,
                    normalize=True, r6d=True, loss_type="mse", use_joint_loss=False, mkdir=False, upper_body_only=True)
    AMASS_IMU2Leaf_model = get_model(AMASS_IMU2Leaf_config).load_from_checkpoint(best_ckpts[model_names[0]], config=AMASS_IMU2Leaf_config)

    AMASS_Leaf2Full_config = Config(experiment=experiment_names, model=model_names[1], project_root_dir=".", joints_set=joint_set.WheelPoser, pred_joints_set=joint_set.upper_body,
                    normalize=True, r6d=True, loss_type="mse", use_joint_loss=False, mkdir=False, upper_body_only=True)
    AMASS_Leaf2Full_model = get_model(AMASS_Leaf2Full_config).load_from_checkpoint(best_ckpts[model_names[1]], config=AMASS_Leaf2Full_config)

    AMASS_Full2Pose_config = Config(experiment=experiment_names, model=model_names[2], project_root_dir=".", joints_set=joint_set.WheelPoser, pred_joints_set=joint_set.upper_body,
                    normalize=True, r6d=True, loss_type="mse", use_joint_loss=False, mkdir=False, upper_body_only=True)
    AMASS_Full2Pose_model = get_model(AMASS_Full2Pose_config).load_from_checkpoint(best_ckpts[model_names[2]], config=AMASS_Full2Pose_config)

    # %%
    # load model

    model_names = ["IMU2Leaf_WheelPoser_WHEELPOSER", "Leaf2Full_WheelPoser_WHEELPOSER", "Full2Pose_WheelPoser_WHEELPOSER"]
    experiment_names = "TransPose_Style_500"

    # %%
    best_ckpts = get_checkpoints(model_names, experiment_names, leave_one_out=leave_one_out)
    print(best_ckpts)

    WHEELPOSER_IMU2Leaf_config = Config(experiment=experiment_names, model=model_names[0], project_root_dir=".", joints_set=joint_set.WheelPoser, pred_joints_set=joint_set.upper_body,
                    normalize=True, r6d=True, loss_type="mse", use_joint_loss=False, mkdir=False, upper_body_only=True, exp_setup='am_only', upsample_copies=7)
    WHEELPOSER_IMU2Leaf_model = get_model(WHEELPOSER_IMU2Leaf_config, pretrained=AMASS_IMU2Leaf_model).load_from_checkpoint(best_ckpts[model_names[0]], config=WHEELPOSER_IMU2Leaf_config, pretrained_model = AMASS_IMU2Leaf_model)

    WHEELPOSER_Leaf2Full_config = Config(experiment=experiment_names, model=model_names[1], project_root_dir=".", joints_set=joint_set.WheelPoser, pred_joints_set=joint_set.upper_body,
                    normalize=True, r6d=True, loss_type="mse", use_joint_loss=False, mkdir=False, upper_body_only=True, exp_setup='am_only', upsample_copies=7)
    WHEELPOSER_Leaf2Full_model = get_model(WHEELPOSER_Leaf2Full_config, pretrained=AMASS_Leaf2Full_model).load_from_checkpoint(best_ckpts[model_names[1]], config=WHEELPOSER_Leaf2Full_config, pretrained_model = AMASS_Leaf2Full_model)

    WHEELPOSER_Full2Pose_config = Config(experiment=experiment_names, model=model_names[2], project_root_dir=".", joints_set=joint_set.WheelPoser, pred_joints_set=joint_set.upper_body,
                    normalize=True, r6d=True, loss_type="mse", use_joint_loss=False, mkdir=False, upper_body_only=True, exp_setup='am_only', upsample_copies=7)
    WHEELPOSER_Full2Pose_model = get_model(WHEELPOSER_Full2Pose_config, pretrained=AMASS_Full2Pose_model).load_from_checkpoint(best_ckpts[model_names[2]], config=WHEELPOSER_Full2Pose_config, pretrained_model = AMASS_Full2Pose_model)

    #modify config in each script
    shared_input_config = WHEELPOSER_IMU2Leaf_config
    shared_output_config = WHEELPOSER_Full2Pose_config



    wheelposer = TransPose_Global_WheelPoser(config=shared_input_config, imu2leaf=WHEELPOSER_IMU2Leaf_model, leaf2full=WHEELPOSER_Leaf2Full_model, full2pose=WHEELPOSER_Full2Pose_model, num_past_frame=num_past_frame, num_future_frame=num_future_frame, physics=physics).to(shared_input_config.device)
    print(f"the actual device is {shared_input_config.device}")
    print(f"wheelposerimu2leaf device is {WHEELPOSER_IMU2Leaf_model.device}")
    print(f"wheelposerleaf2full device is {WHEELPOSER_Leaf2Full_model.device}")
    print(f"wheelposerfull2pose device is {WHEELPOSER_Full2Pose_model.device}")

    #Setup the IMU
    imu_set = IMUSet(buffer_len=1)
    imu_set.establish_connection()
    clock = Clock()
    print('Finished setting up.')

    time.sleep(3)
    print('Check heading reset')
    oris = imu_set.get_mean_measurement_of_n_second(num_seconds=1, buffer_len=80)[0]

    for i in range (imu_count):
        print(quaternion_to_axis_angle(oris[i]))


    input('Put imu leftarm aligned with your body reference frame (x = Left, y = Up, z = Forward) and then press any key.')
    # listener = keyboard.Listener(on_press=on_press)
    # listener.start()
    for i in range(3, 0, -1):
        print('\rHold the imu stably and be ready. The celebration will begin after %d seconds.' % i, end='')
        time.sleep(1)
    print('Keep for 3 seconds ...', end='')
    oris = imu_set.get_mean_measurement_of_n_second(num_seconds=3, buffer_len=200)[0][0]
    smpl2imu = quaternion_to_rotation_matrix(oris).view(3, 3).t() # global to sensor frame

    print(oris)
    print(smpl2imu)

    input('\tFinish.\nWear all imus correctly and press any key.')
    for i in range(3, 0, -1):
        print('\rStand straight in T-pose and be ready. The celebration will begin after %d seconds.' % i, end='')
        time.sleep(1)
    print('\rStand straight in T-pose. Keep the pose for 3 seconds ...', end='')
    oris, accs = imu_set.get_mean_measurement_of_n_second(num_seconds=3, buffer_len=200)
    oris = quaternion_to_rotation_matrix(oris) #sensor to global
    device2bone = smpl2imu.matmul(oris).transpose(1, 2).matmul(torch.eye(3))
    acc_offsets = smpl2imu.matmul(accs.unsqueeze(-1))   # [num_imus, 3, 1], already in global inertial frame

    # move matrices to device
    smpl2imu = smpl2imu.to(shared_input_config.device)
    device2bone = device2bone.to(shared_input_config.device)
    acc_offsets = acc_offsets.to(shared_input_config.device)

    print(device2bone)
    print(acc_offsets)

    print('\tFinish.\nStart estimating poses.')

    imu_set.start_reading()
    
    if unity_visualizer:
        server_for_unity = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_for_unity.bind((server_unity_ip, server_unity_port))
        server_for_unity.listen(5)
        print('Server start. Waiting for unity3d to connect.')
        conn, addr_unity = server_for_unity.accept()
        print ('Got connection from', addr_unity )

    running = True
    clock = Clock()
    old_timestamp = 0
    is_recording = False
    record_buffer = None

    get_input_thread = threading.Thread(target=get_input)
    get_input_thread.setDaemon(True)
    get_input_thread.start()
    wheelposer.eval()
    print(wheelposer.device)
    print(wheelposer.device)
    print(wheelposer.device)
    print(wheelposer.device)

    pygame.init()
    while running:
        start_time = time.time()
        # print(imu_set.clock.get_fps(), clock.get_fps())
        ori_raw, acc_raw = imu_set.get_current_buffer()   # [1, 4, 4], get measurements in running fps
        # move to device
        ori_raw = ori_raw.to(shared_input_config.device)
        acc_raw = acc_raw.to(shared_input_config.device)

        if (ori_raw.size(0)!=0) and (acc_raw.size(0)!=0) and (old_timestamp!=imu_set._packet_count):
            old_timestamp = imu_set._packet_count
            # calibration
            ori_raw = quaternion_to_rotation_matrix(ori_raw).view(1, 4, 3, 3)
            acc_cal = (smpl2imu.matmul(acc_raw.view(-1, 4, 3, 1)) - acc_offsets).view(1, 4, 3)
            ori_cal = smpl2imu.matmul(ori_raw).matmul(device2bone)
            imu_recording = torch.cat((acc_cal.view(-1,12), ori_cal.view(-1,36)), dim=1)

            # normalization
            acc = torch.cat((acc_cal[:, :3] - acc_cal[:, 3:], acc_cal[:, 3:]), dim=1).bmm(ori_cal[:, -1]) / WHEELPOSER_Full2Pose_config.acc_scale
            ori = torch.cat((ori_cal[:, 3:].transpose(2, 3).matmul(ori_cal[:, :3]), ori_cal[:, 3:]), dim=1)
            # data_nn = torch.cat((acc.view(-1, 12), ori.view(-1, 36)), dim=1).to(shared_input_config.device)
            data_nn = torch.cat((acc.view(-1, 12), ori.view(-1, 36)), dim=1)

            # prediction
            pose = wheelposer.forward_online(data_nn)
            # pose = rotation_matrix_to_axis_angle(pred_pose).view(72)
            tran = torch.tensor([0,-0.4,-0.1055]).to(device)


            # recording
            if not is_recording and start_recording:
                record_buffer = imu_recording.view(1, -1)
                is_recording = True
                start_time = time.time()
            elif is_recording and start_recording:
                record_buffer = torch.cat([record_buffer, imu_recording.view(1, -1)], dim=0)
            elif is_recording and not start_recording:
                end_time = time.time()
                torch.save(record_buffer, 'src/data/imu_recordings/r' + datetime.now().strftime('%T').replace(':', '-') + '.pt')
                is_recording = False
                recording_time = end_time-start_time
                fps = record_buffer.size(0)/recording_time
                print('Recording FPS is: ', fps)

            s = ','.join(['%g' % v for v in pose]) + '#' + \
            ','.join(['%g' % v for v in tran]) + '$'

            if unity_visualizer:
                conn.send(s.encode('utf8'))  # I use unity3d to read pose and translation for visualization here
            # clock.tick(60)
            # print(timestamp)
            print(f"FPS: {1/(time.time()-start_time)}")
            # clock.tick(60)
            # print(s)