import argparse
import torch
import pickle as pkl
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from src.config import Config, joint_set
from src.evaluator import ReducedPoseEvaluator, WheelPoserEvaluator
from src.models.utils import get_model
from src.data.utils import get_datamodule, get_dataset
from src.articulate.math import *
from src.utils import get_checkpoints, reduced_upperbody_6d_to_full_local_mat
from src.models.LSTMs.Three_Stage_Global.Three_Stage_Global_WheelPoser_Wrapper import Three_Stage_Global_WheelPoser

import torch
from pathlib import Path
from datetime import datetime
import numpy as np
import shutil 


num_past_frame = 20
num_future_frame = 5
physics = True


seed_everything(42, workers=True)


model_names = ["IMU2Leaf_WheelPoser_AMASS", "Leaf2Full_WheelPoser_AMASS", "Full2Pose_WheelPoser_AMASS"]
experiment_names = "3_Stage_500"
leave_one_out = '14'

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
experiment_names = "3_Stage_500"

# %%
best_ckpts = get_checkpoints(model_names, experiment_names, leave_one_out=leave_one_out)
print(best_ckpts)

WHEELPOSER_IMU2Leaf_config = Config(experiment=experiment_names, model=model_names[0], project_root_dir=".", joints_set=joint_set.WheelPoser, pred_joints_set=joint_set.upper_body,
                normalize=True, r6d=True, loss_type="mse", use_joint_loss=False, mkdir=False, upper_body_only=True, exp_setup='leave_14_out', upsample_copies=7)
WHEELPOSER_IMU2Leaf_model = get_model(WHEELPOSER_IMU2Leaf_config, pretrained=AMASS_IMU2Leaf_model).load_from_checkpoint(best_ckpts[model_names[0]], config=WHEELPOSER_IMU2Leaf_config, pretrained_model = AMASS_IMU2Leaf_model)

WHEELPOSER_Leaf2Full_config = Config(experiment=experiment_names, model=model_names[1], project_root_dir=".", joints_set=joint_set.WheelPoser, pred_joints_set=joint_set.upper_body,
                normalize=True, r6d=True, loss_type="mse", use_joint_loss=False, mkdir=False, upper_body_only=True, exp_setup='leave_14_out', upsample_copies=7)
WHEELPOSER_Leaf2Full_model = get_model(WHEELPOSER_Leaf2Full_config, pretrained=AMASS_Leaf2Full_model).load_from_checkpoint(best_ckpts[model_names[1]], config=WHEELPOSER_Leaf2Full_config, pretrained_model = AMASS_Leaf2Full_model)

WHEELPOSER_Full2Pose_config = Config(experiment=experiment_names, model=model_names[2], project_root_dir=".", joints_set=joint_set.WheelPoser, pred_joints_set=joint_set.upper_body,
                normalize=True, r6d=True, loss_type="mse", use_joint_loss=False, mkdir=False, upper_body_only=True, exp_setup='leave_14_out', upsample_copies=7)
WHEELPOSER_Full2Pose_model = get_model(WHEELPOSER_Full2Pose_config, pretrained=AMASS_Full2Pose_model).load_from_checkpoint(best_ckpts[model_names[2]], config=WHEELPOSER_Full2Pose_config, pretrained_model = AMASS_Full2Pose_model)

#modify config in each script
shared_input_config = WHEELPOSER_IMU2Leaf_config
shared_output_config = WHEELPOSER_Full2Pose_config



wheelposer = Three_Stage_Global_WheelPoser(config=shared_input_config, imu2leaf=WHEELPOSER_IMU2Leaf_model, leaf2full=WHEELPOSER_Leaf2Full_model, full2pose=WHEELPOSER_Full2Pose_model, num_past_frame=num_past_frame, num_future_frame=num_future_frame, physics=physics).to(shared_input_config.device)

# get the data
_, input_test_dataset, _ = get_dataset(shared_input_config)
_, output_test_dataset, _ = get_dataset(shared_output_config)

input_test_dataset[0][0].shape, output_test_dataset[0][1].shape, len(input_test_dataset), len(output_test_dataset)

preds, trues = [], []

AMASS_IMU2Leaf_model.eval()
AMASS_Leaf2Full_model.eval()
AMASS_Full2Pose_model.eval()
WHEELPOSER_IMU2Leaf_model.eval()
WHEELPOSER_Leaf2Full_model.eval()
WHEELPOSER_Full2Pose_model.eval()

forces = []

with torch.no_grad():
    for i in range(len(input_test_dataset)):
        print(i, len(input_test_dataset))
        # imu_input = input_test_dataset[i][0].to(shared_input_config.device)
        # pred_pose = wheelposer.forward_offline(imu_input)
        imu_input = input_test_dataset[i][0].to(shared_input_config.device)
        # online_results = [wheelposer.forward_online(f) for f in torch.cat((imu_input, imu_input[-1].repeat(num_future_frame, 1)))]
        # pred_pose = torch.stack(online_results[num_future_frame:])
        # preds.append(pred_pose)
        # trues.append(output_test_dataset.pose[i].reshape(-1, 24, 3, 3).to('cpu'))
        online_results_pose = []
        online_results_tau = []
        for f in torch.cat((imu_input, imu_input[-1].repeat(num_future_frame, 1))):
            online_results = wheelposer.forward_online(f)   
            online_results_pose.append(online_results[0])
            online_results_tau.append(online_results[1])
        pred_pose = torch.stack(online_results_pose[num_future_frame:])
        preds.append(pred_pose)
        trues.append(output_test_dataset.pose[i].reshape(-1, 24, 3, 3).to('cpu'))
        forces.append(torch.stack(online_results_tau[num_future_frame:]))

# +
# imu_input.shape, pred_leaf.shape, _input.shape, pred_full.shape, pred_pose.shape
# -
preds_tensor = torch.cat(preds)
trues_tensor = torch.cat(trues)

preds_m = preds_tensor[:, joint_set.upper_body]
trues_m = trues_tensor[:, joint_set.upper_body]

preds_m.shape, trues_m.shape

preds_aa = rotation_matrix_to_axis_angle(preds_m).view(preds_tensor.shape[0], -1)
trues_aa = rotation_matrix_to_axis_angle(trues_m).view(trues_tensor.shape[0], -1)

# preds_tensor = torch.cat(preds)
# trues_tensor = torch.cat(trues)

# preds_aa = rotation_matrix_to_axis_angle(preds_tensor).view(preds_tensor.shape[0], -1)
# trues_aa = rotation_matrix_to_axis_angle(trues_tensor).view(trues_tensor.shape[0], -1)

evaluator = WheelPoserEvaluator(shared_input_config)

# +
chunk_size = 1000

metrics = []
for ps, ts in zip(preds_m.split(chunk_size), trues_m.split(chunk_size)):
    metrics.append(evaluator(ps, ts))
    
avg_metrics = torch.stack(metrics).mean(dim=0)
total_metrics = torch.stack(metrics).sum(dim=0)
# -

for i, name in enumerate(evaluator.names):
    if 'Travel' in name:
        print(f"{name}: {total_metrics[i]}")
    else:
        print(f"{name}: {avg_metrics[i]}")

# +
# save preds_aa, trues_aa
import pickle as pkl

# save the preds and trues
with open(f"pred_results/{experiment_names}_{leave_one_out}_end2end_results.pkl", "wb") as f:
    dump = {
        "p": preds_aa.cpu().numpy(),
        "t": trues_aa.cpu().numpy()
    }
    
    pkl.dump(dump, f)