import argparse
import torch
import pickle as pkl
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from src.config import Config, joint_set, paths
from src.evaluator import ReducedPoseEvaluator, WheelPoserEvaluator
# from src.models.utils import get_model
# from src.data.utils import get_datamodule, get_dataset
from src.articulate.math import *
from src import articulate as art

import torch
from pathlib import Path
from datetime import datetime
import numpy as np
import shutil
import enum
import torch
import numpy as np
# import pybullet as p
# from src.articulate.math import rotation_matrix_to_euler_angle_np, euler_angle_to_rotation_matrix_np, euler_convert_np, \
#     normalize_angle

def get_checkpoints(model_name: list, experiment_names: str, leave_one_out=None):
    path_to_checkpoints = Path("checkpoints")
    
    checkpoints = [x.name for x in path_to_checkpoints.iterdir() if experiment_names in x.name]

    best_ckpts = {}


    for model_name in model_name:
        filtered_checkpoints = []
        for checkpoint in checkpoints:
            if model_name == checkpoint.split("-",2)[1]:
                if leave_one_out!=None and 'WHEELPOSER' in model_name:
                    if leave_one_out in checkpoint.split("-",3)[2]:
                        filtered_checkpoints.append(checkpoint)
                else:
                    filtered_checkpoints.append(checkpoint)
        # get the latest model_checkpoint
        if leave_one_out!=None and 'WHEELPOSER' in model_name:
            model_creation_dates = [datetime.strptime(x.split("-", 3)[3], "%m%d%Y-%H%M%S") for x in filtered_checkpoints]
        else:
            model_creation_dates = [datetime.strptime(x.split("-", 2)[2], "%m%d%Y-%H%M%S") for x in filtered_checkpoints]

        latest_model = filtered_checkpoints[np.argmax(model_creation_dates)]

        # now get the best ckpt
        try:
            with open(path_to_checkpoints / latest_model / "best_model.txt", "r") as f:
                best_model_name = Path(f.readlines()[0].strip()).name
            
            best_ckpts[model_name] = path_to_checkpoints / latest_model / best_model_name
        except:
            print("best_model.txt is missing, using the latest checkpoint")
            ckpts = [(x.name.split("=")[2].split("-")[0], x.name) for x in (path_to_checkpoints / latest_model).iterdir() if "epoch" in x.name]
            best_ckpt = sorted(ckpts, key=lambda x: x[0])[-1][1]
            best_ckpts[model_name] = path_to_checkpoints / latest_model / best_ckpt
    
    return best_ckpts


def reduced_6d_to_full_local_mat(root_rotation, glb_reduced_pose):
    body_model = art.ParametricModel(paths.smpl_file)
    glb_reduced_pose = r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
    global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
    global_full_pose[:, joint_set.reduced] = glb_reduced_pose
    pose = body_model.inverse_kinematics_R(global_full_pose).view(-1, 24, 3, 3)
    # pose = global_full_pose.view(-1, 24, 3, 3)
    pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
    pose[:, 0] = root_rotation.view(-1, 3, 3)
    return pose

def reduced_upperbody_6d_to_full_local_mat(root_rotation, glb_reduced_pose):
    body_model = art.ParametricModel(paths.smpl_file)
    glb_reduced_pose = r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced_upper_body, 3, 3)
    global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
    global_full_pose[:, joint_set.reduced_upper_body] = glb_reduced_pose
    pose = body_model.inverse_kinematics_R(global_full_pose).view(-1, 24, 3, 3)
    # pose = global_full_pose.view(-1, 24, 3, 3)
    pose[:, joint_set.ignored_upper_body] = torch.eye(3, device=pose.device)
    pose[:, 0] = root_rotation.view(-1, 3, 3)
    return pose



class Body(enum.Enum):
    r"""
    Prefix L = left; Prefix R = right.
    """
    ROOT = 2
    PELVIS = 2
    SPINE = 2
    LHIP = 5
    RHIP = 17
    SPINE1 = 29
    LKNEE = 8
    RKNEE = 20
    SPINE2 = 32
    LANKLE = 11
    RANKLE = 23
    SPINE3 = 35
    LFOOT = 14
    RFOOT = 26
    NECK = 68
    LCLAVICLE = 38
    RCLAVICLE = 53
    HEAD = 71
    LSHOULDER = 41
    RSHOULDER = 56
    LELBOW = 44
    RELBOW = 59
    LWRIST = 47
    RWRIST = 62
    LHAND = 50
    RHAND = 65



_smpl_to_rbdl = [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 28, 29, 3, 4, 5, 12, 13, 14, 21, 22, 23, 30, 31, 32, 6, 7, 8,
                 15, 16, 17, 24, 25, 26, 36, 37, 38, 45, 46, 47, 51, 52, 53, 57, 58, 59, 63, 64, 65, 39, 40, 41,
                 48, 49, 50, 54, 55, 56, 60, 61, 62, 66, 67, 68, 33, 34, 35, 42, 43, 44]
_rbdl_to_smpl = [0, 1, 2, 12, 13, 14, 24, 25, 26, 3, 4, 5, 15, 16, 17, 27, 28, 29, 6, 7, 8, 18, 19, 20, 30, 31, 32,
                 9, 10, 11, 21, 22, 23, 63, 64, 65, 33, 34, 35, 48, 49, 50, 66, 67, 68, 36, 37, 38, 51, 52, 53, 39,
                 40, 41, 54, 55, 56, 42, 43, 44, 57, 58, 59, 45, 46, 47, 60, 61, 62]
_rbdl_to_bullet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                   27, 28, 29, 30, 31, 32, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 33, 34, 35,
                   36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 63, 64, 65, 66, 67, 68]
smpl_to_rbdl_data = _smpl_to_rbdl


# def set_pose(id_robot, q):
#     r"""
#     Set the robot configuration.
#     """
#     p.resetJointStatesMultiDof(id_robot, list(range(1, p.getNumJoints(id_robot))), q[6:][_rbdl_to_bullet].reshape(-1, 1))
#     glb_rot = p.getQuaternionFromEuler(euler_convert_np(q[3:6], 'zyx', 'xyz')[[2, 1, 0]])
#     p.resetBasePositionAndOrientation(id_robot, q[:3], glb_rot)


def smpl_to_rbdl(poses, trans):
    r"""
    Convert smpl poses and translations to robot configuration q. (numpy, batch)

    :param poses: Array that can reshape to [n, 24, 3, 3].
    :param trans: Array that can reshape to [n, 3].
    :return: Ndarray in shape [n, 75] (3 root position + 72 joint rotation).
    """
    poses = np.array(poses).reshape(-1, 24, 3, 3)
    trans = np.array(trans).reshape(-1, 3)
    euler_poses = rotation_matrix_to_euler_angle_np(poses[:, 1:], 'XYZ').reshape(-1, 69)
    euler_glbrots = rotation_matrix_to_euler_angle_np(poses[:, :1], 'xyz').reshape(-1, 3)
    euler_glbrots = euler_convert_np(euler_glbrots[:, [2, 1, 0]], 'xyz', 'zyx')
    qs = np.concatenate((trans, euler_glbrots, euler_poses[:, _smpl_to_rbdl]), axis=1)
    qs[:, 3:] = normalize_angle(qs[:, 3:])
    return qs


def rbdl_to_smpl(qs):
    r"""
    Convert robot configuration q to smpl poses and translations. (numpy, batch)

    :param qs: Ndarray that can reshape to [n, 75] (3 root position + 72 joint rotation).
    :return: Poses ndarray in shape [n, 24, 3, 3] and translation ndarray in shape [n, 3].
    """
    qs = qs.reshape(-1, 75)
    trans, euler_glbrots, euler_poses = qs[:, :3], qs[:, 3:6], qs[:, 6:][:, _rbdl_to_smpl]
    euler_glbrots = euler_convert_np(euler_glbrots, 'zyx', 'xyz')[:, [2, 1, 0]]
    glbrots = euler_angle_to_rotation_matrix_np(euler_glbrots, 'xyz').reshape(-1, 1, 3, 3)
    poses = euler_angle_to_rotation_matrix_np(euler_poses, 'XYZ').reshape(-1, 23, 3, 3)
    poses = np.concatenate((glbrots, poses), axis=1)
    return poses, trans