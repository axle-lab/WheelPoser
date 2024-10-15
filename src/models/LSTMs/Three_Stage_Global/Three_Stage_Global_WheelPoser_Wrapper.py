import torch.nn as nn
import torch
import pytorch_lightning as pl
from src.models.LSTMs.RNN import RNN
from src.models.loss_functions import *
from src.config import *
from src.utils import *
# from src.physics.dynamics import *

class Three_Stage_Global_WheelPoser(pl.LightningModule):
    r"""
    3 Stage Pose S1
    Inputs - N IMUs, Outputs - SMPL Pose params (in Rot Matrix)
    """
    def __init__(self, config, imu2leaf, leaf2full, full2pose, num_past_frame = 20, num_future_frame = 5, physics=False):
        super().__init__()
        n_input = 12 * len(config.joints_set)
        self.batch_size = config.batch_size
        self.pose_s1 = imu2leaf
        self.pose_s2 = leaf2full
        self.pose_s3 = full2pose

        self.num_past_frame = num_past_frame
        self.num_future_frame = num_future_frame
        self.num_total_frame = num_past_frame + num_future_frame + 1

        self.physics = physics
        
        # self.dynamics_optimizer = PhysicsOptimizer(debug=False)
        self.imu = None
        self.eval()

    def forward(self, imu_input):
        pred_leaf = self.pose_s1(imu_input.unsqueeze(0), [imu_input.shape[0]]).squeeze(0)
        _input = torch.cat([pred_leaf, imu_input], dim=1)
        pred_full = self.pose_s2(_input.unsqueeze(0), [_input.shape[0]]).squeeze(0)
        _input = torch.cat([pred_full, imu_input], dim=1)
        pred_pose = self.pose_s3(_input.unsqueeze(0), [_input.shape[0]]).squeeze(0)
        return pred_pose

    @torch.no_grad()
    def forward_online(self, imu_inputs):
        imu = imu_inputs.repeat(self.num_total_frame, 1) if self.imu is None else torch.cat((self.imu[1:], imu_inputs.view(1, -1)))
        pred_pose = self.forward(imu)
        #get current frame
        pred_pose = pred_pose[self.num_past_frame]

        ## Vimal commenting from here TODO uncomment
        preds_m = r6d_to_rotation_matrix(pred_pose) # the shape of this is 16, 3, 3

        # fill in the rest of the joints with identity
        # upper_body = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        # pred_pose = torch.zeros(24, 3, 3).to(self.device)

        # pred_pose[upper_body] = preds_m


        preds_aa = rotation_matrix_to_axis_angle(preds_m).view(48) # runs on cpu
        pelvis = preds_aa[:3]
        pelvis_rotation_matrix = axis_angle_to_rotation_matrix(pelvis).view(3,3)
        pelvis_y = pelvis_rotation_matrix[1]
        world_y = torch.tensor([0,1,0], dtype=torch.float32).to(self.device)
        angle_between = torch.acos(torch.dot(pelvis_y, world_y)/(torch.norm(pelvis_y)*torch.norm(world_y)))
        angle_between = 90 - angle_between * 180 / np.pi
        hip_angle = (-angle_between) * np.pi / 180
        # hip_angle = -1.53 - preds_aa[0]
        hip_adjustment = torch.tensor([hip_angle,0,0]).to(self.device) #in radians
        knee_adjustment = torch.tensor([1.46,0,0]).to(self.device) #in radians
        zeros = torch.zeros(3).to(self.device)
        pose = torch.cat((preds_aa[:3], hip_adjustment, hip_adjustment, preds_aa[3:6], 
                                knee_adjustment, knee_adjustment, preds_aa[6:9], zeros, zeros, 
                                preds_aa[9:12], zeros, zeros, preds_aa[12:]), dim=0)
        pred_pose = pose.view(72)
        # pred_pose = axis_angle_to_rotation_matrix(pose).view(24,3,3)
        #############
        
        self.imu = imu
        # if self.physics:
        #     root_ori = pred_pose[0]
        #     pred_pose[0] = torch.eye(3)
        #     pred_pose_optimized, tau = self.dynamics_optimizer.optimize_frame(pred_pose)
        #     pred_pose_optimized[0] = root_ori
        #     return pred_pose_optimized, tau
        return pred_pose
    
    @torch.no_grad()
    def forward_offline(self, imu_inputs):
        pred_pose = self.forward(imu_inputs)
        preds_m = r6d_to_rotation_matrix(pred_pose)
        preds_aa = rotation_matrix_to_axis_angle(preds_m).view(-1, 48)
        hip_angle = -1.53 - preds_aa[:, 0]
        # hip_adjustment = torch.tensor([hip_angle,0,0]).to(self.device)
        hip_adjustment = torch.zeros(hip_angle.shape[0], 3).to(self.device)
        hip_adjustment = torch.cat((hip_angle.unsqueeze(1), hip_adjustment[:, 1:]), dim=1).to(self.device)
        # hip_adjustment = torch.tensor([-1.4,0,0]).repeat(preds_aa.shape[0],1).to(self.device) #in radians
        knee_adjustment = torch.tensor([1.46,0,0]).repeat(preds_aa.shape[0],1).to(self.device) #in radians
        zeros = torch.zeros(3).repeat(preds_aa.shape[0],1).to(self.device)
        pose = torch.cat((preds_aa[:, :3], hip_adjustment, hip_adjustment, preds_aa[:, 3:6], 
                                knee_adjustment, knee_adjustment, preds_aa[:, 6:9], zeros, zeros, 
                                preds_aa[:, 9:12], zeros, zeros, preds_aa[:, 12:]), dim=1)
        pred_pose = axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3).to('cpu')
        # if self.physics:
        #     pred_pose_optimized = [self.dynamics_optimizer.optimize_frame(pred_pose[i]) for i in range(pred_pose.shape[0])]
        #     pred_pose_optimized = torch.stack(pred_pose_optimized)
        #     return pred_pose_optimized
        
        return pred_pose
    
    # def reset_physics(self):
    #     self.dynamics_optimizer.reset_states()

