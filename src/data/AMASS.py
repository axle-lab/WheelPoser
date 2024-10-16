import torch
from torch.utils.data import Dataset
from src.articulate import math
from src.config import leaf_joints
from src.config import Config, joint_set
from src.articulate.model import*
from src import articulate as art
from src.config import paths

class AMASS(Dataset):
    def __init__(self, split="train", config=None, stage="imu2leaf"):
        super().__init__()

        # load the data
        self.train = split
        self.config = config
        self.data = self.load_data(self.config.normalize)
        self.body_model = art.ParametricModel(paths.smpl_file)

        self.stage = stage
        
    def load_data(self, normalize=False):
        def _syn_vel(v):
            r"""
            Synthesize velocities from vertex positions.
            """
            vel = torch.stack([(v[i+1] - v[i]) * 60 for i in range(0, v.shape[0] - 1)])
            vel = torch.cat((torch.zeros_like(vel[:1]), vel))
            return vel

        if self.train == "train":
            data_files = [x.name for x in self.config.processed_wheelposer_amass_nn_ready_4.iterdir()]
        else:
            data_files = ['wheelposer_wu_fullset.pt']

        imu = []
        leaf_joint_pos = []
        full_joint_pos = []
        pose = []
        if self.config.use_vel_loss == True:
            vel = []

        num_joints = len(self.config.joints_set)
        for fname in data_files:
            if self.train == "train":
                fdata = torch.load(self.config.processed_wheelposer_amass_nn_ready_4 / fname)
            else:
                fdata = torch.load(self.config.processed_wheelposer_nn_ready_4 / fname)

            for i in range(len(fdata["acc"])):
                # inputs
                facc = fdata["acc"][i] 
                fori = fdata["ori"][i]

                glb_acc = (facc[:, self.config.joints_set]).view(-1, num_joints, 3)
                glb_ori = (fori[:, self.config.joints_set]).view(-1, num_joints, 3, 3)

                if normalize == "no_translation":
                    acc = glb_acc
                    ori = glb_ori
                elif normalize == True:
                    # make acc relative to the last imu always
                    acc = torch.cat((glb_acc[:, :num_joints-1] - glb_acc[:, num_joints-1:], glb_acc[:, num_joints-1:]), dim=1).bmm(glb_ori[:, -1]) / self.config.acc_scale
                    ori = torch.cat((glb_ori[:, num_joints-1:].transpose(2, 3).matmul(glb_ori[:, :num_joints-1]), glb_ori[:, num_joints-1:]), dim=1)
                else:
                    acc = glb_acc
                    ori = glb_ori

                imu_inputs = torch.cat([acc.flatten(1), ori.flatten(1)], dim=1)
                
                # outputs
                fjoint = fdata["joint"][i]

                # Synthesize vel
                if self.config.use_vel_loss == True:
                    glb_vel = _syn_vel(fjoint)

                if normalize == "no_translation":
                    fjoint_leaf = fjoint[:, leaf_joints] - fjoint[:, [0]] # pelvis is idx 0 in the joints
                    fjoint_full = fjoint[:, self.config.pred_joints_set[1:]] - fjoint[:, [0]] # pelvis is idx 0 in the joints

                    if self.config.use_vel_loss == True:
                        fvel_full = glb_vel[:, self.config.pred_joints_set[1:]] - glb_vel[:, [0]] # pelvis is idx 0 in the joints
                elif normalize == True:
                    fjoint_leaf = (fjoint[:, leaf_joints] - fjoint[:, [0]]).bmm(glb_ori[:, -1]) # make it root relative
                    fjoint_full = (fjoint[:, self.config.pred_joints_set[1:]] - fjoint[:, [0]]).bmm(glb_ori[:, -1])

                    if self.config.use_vel_loss == True:
                        fvel_full = glb_vel[:, self.config.pred_joints_set[1:]] - glb_vel[:, [0]] # pelvis is idx 0 in the joints
                else:
                    fjoint_leaf = fjoint[:, leaf_joints]
                    fjoint_full = fjoint
                    if self.config.use_vel_loss == True:
                        fvel_full = glb_vel

                fjoint_leaf = fjoint_leaf.reshape(fjoint_leaf.shape[0], -1)
                fjoint_full = fjoint_full.reshape(fjoint_full.shape[0], -1)

                #add Gaussian noise to the joint positions
                noise_leaf = torch.randn(fjoint_leaf.shape) * 0.04
                fjoint_leaf += noise_leaf

                noise_full = torch.randn(fjoint_full.shape) * 0.025
                fjoint_full += noise_full

                fpose = fdata["pose"][i]
                fpose = fpose.reshape(fpose.shape[0], -1)

                if self.config.use_vel_loss == True:
                    fvel_full = fvel_full.flatten(1)

                window_length = self.config.max_sample_len
                if "Transformer" in self.config.model:
                    # Transformers can't handle batches of different input lengths, so we only keep windows that are the proper length
                    imu.extend([x for x in torch.split(imu_inputs, window_length) if x.shape[0] == window_length])
                    leaf_joint_pos.extend([x for x in torch.split(fjoint_leaf, window_length) if x.shape[0] == window_length])
                    full_joint_pos.extend([x for x in torch.split(fjoint_full, window_length) if x.shape[0] == window_length])
                    pose.extend([x for x in torch.split(fpose, window_length) if x.shape[0] == window_length])
                elif self.train == "test":
                    imu.extend([imu_inputs])
                    leaf_joint_pos.extend([fjoint_leaf])
                    full_joint_pos.extend([fjoint_full])
                    pose.extend([fpose])
                    if self.config.use_vel_loss == True:
                        vel.extend([fvel_full])
                else:
                    imu.extend(torch.split(imu_inputs, window_length))
                    leaf_joint_pos.extend(torch.split(fjoint_leaf, window_length))
                    full_joint_pos.extend(torch.split(fjoint_full, window_length))
                    pose.extend(torch.split(fpose, window_length))
                    if self.config.use_vel_loss == True:
                        vel.extend(torch.split(fvel_full, window_length))

        self.imu = imu
        self.leaf_joint_pos = leaf_joint_pos
        self.full_joint_pos = full_joint_pos
        self.pose = pose

        if self.config.use_vel_loss == True:
            self.vel = vel

    def __getitem__(self, idx):
        _imu = self.imu[idx].float()
        _leaf_joints = self.leaf_joint_pos[idx].float()
        _full_joints = self.full_joint_pos[idx].float()
        _pose = self.pose[idx].float()
        if self.config.use_vel_loss == True:
            _vel = self.vel[idx].float()

        if self.stage == "imu2leaf":
            _input = _imu
            _output = _leaf_joints
        elif self.stage == "leaf2full":
            _input = torch.concat((_leaf_joints, _imu), dim=1)
            _output = _full_joints
        elif self.stage == "full2pose":
            _input = torch.concat((_full_joints, _imu), dim=1)
            if self.config.r6d == True:
                if self.config.reduced_pose_output == True:
                    #get global joint rotations
                    glb_joint_rotations, _ = self.body_model.forward_kinematics(_pose)
                    root_relative_joint_rotations = torch.matmul(glb_joint_rotations[:, 0].unsqueeze(1).transpose(2,3), glb_joint_rotations)
                    _output = math.rotation_matrix_to_r6d(root_relative_joint_rotations).reshape(-1, 24, 6)[:, joint_set.reduced_upper_body].reshape(-1, 6 * len(joint_set.reduced_upper_body))
                else:
                    _output = math.rotation_matrix_to_r6d(_pose).reshape(-1, 24, 6)[:, self.config.pred_joints_set].reshape(-1, 6 * len(self.config.pred_joints_set))
            else:
                _output = _pose.reshape(-1, 24, 3, 3)[:,self.config.pred_joints_set].reshape(-1, 9*len(self.config.pred_joints_set))
        elif self.stage == "full":
            _input = _imu
            if self.config.r6d == True:
                _output = math.rotation_matrix_to_r6d(_pose).reshape(-1, 24, 6)[:, self.config.pred_joints_set].reshape(-1, 6 * len(self.config.pred_joints_set))
            else:
                _output = _pose.reshape(-1, 24, 3, 3)[:,self.config.pred_joints_set].reshape(-1, 9*len(self.config.pred_joints_set))

            if self.config.use_vel_loss == True:
                _output = torch.concat((_output, _vel), dim=1)

        return _input, _output

    def __len__(self):
        return len(self.imu)

