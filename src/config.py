r"""
    Config for paths, joint set, and normalizing scales.
"""

from pathlib import Path
import torch
import datetime


# datasets (directory names) in AMASS
#including GRAB and SOMA
amass_datasets_updated = ['ACCAD', 'BioMotionLab_NTroje', 'BMLhandball', 'BMLmovi', 'CMU',
                  'DanceDB', 'DFaust_67', 'EKUT', 'Eyes_Japan_Dataset', 'GRAB', 'HUMAN4D',
                  'HumanEva', 'KIT', 'MPI_HDM05', 'MPI_Limits', 'MPI_mosh', 'SFU', 'SOMA',
                  'SSM_synced', 'TCD_handMocap', 'TotalCapture', 'Transitions_mocap']

GRAB_SOMA = ['GRAB', 'SOMA']



class paths:
    raw_amass_dir = 'src/data/dataset_raw/AMASS'      # raw AMASS dataset path (raw_amass_dir/ACCAD/ACCAD/s001/*.npz)
    amass_dir = 'src/data/dataset_work/AMASS'         # output path for the synthetic AMASS dataset
    amass_dir_debug = 'src/data/dataset_raw/AMASS_debug'
    raw_wheelposer_dir = 'src/data/dataset_raw/WheelPoser'  # raw WheelPoser dataset path

    example_dir = 'data/example'                    # example IMU measurements
    smpl_file = 'src/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'              # official SMPL model path
    weights_file = 'data/weights.pt'                # network weight file
    physics_model_file = 'src/physics/physics.urdf'      # physics body model path
    plane_file = 'src/physics/plane.urdf'                # (for debug) path to plane.urdf    Please put plane.obj next to it.
    physics_parameter_file = 'src/physics/physics_parameters.json'   # physics hyperparameters

    imu_recordings_dir = 'src/data/imu_recordings'
    mocap_recordings_dir = 'src/data/mocap_recordings'

class joint_set:
    WheelPoser = [0, 1, 2, 3]
    TransPose = [0, 1, 2, 3, 4, 5]
    DIP_WheelPoser = [13, 14, 0, 1]
    leaf = [7, 8, 12, 20, 21]
    full = list(range(1, 24))
    full_with_root = list(range(0, 24))
    reduced = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]
    
    #WheelPoser Modification
    upper_body = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    reduced_upper_body = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored_upper_body = [0, 1, 2, 4, 5, 7, 8, 10, 11, 20, 21, 22, 23]
    wheelposer_leaf = [12, 20, 21]
    wheelposer_full = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    lower_body = [1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]

    n_leaf = len(leaf)
    n_full = len(full)
    n_reduced = len(reduced)
    n_ignored = len(ignored)
    n_reduced_upper_body = len(reduced_upper_body)
    n_wheelposer_leaf = len(wheelposer_leaf)
    n_wheelposer_full = len(wheelposer_full)
    n_upper_body = len(upper_body)



class Config:
    def __init__(self, experiment=None, model=None, project_root_dir=None, joints_set=None, loss_type=None, mkdir=True, normalize=False, r6d=False, device=None, use_joint_loss=False, use_glb_rot_loss=False, use_acc_recon_loss=False, pred_joints_set=None, pred_last_frame=False, use_vposer_loss=False, use_vel_loss=False, upper_body_only = False, reduced_pose_output = False, 
                 mixed_imu_set = False, leave_one_out = False, video_data = False, exp_setup = None, upsample_copies = 7):
        self.experiment = experiment
        self.model = model
        self.root_dir = Path(project_root_dir).absolute()
        self.joints_set = joints_set
        self.pred_joints_set = [*range(24)] if pred_joints_set == None else pred_joints_set

        self.mkdir = mkdir
        self.normalize = normalize
        self.r6d = r6d
        self.use_joint_loss = use_joint_loss
        self.use_glb_rot_loss = use_glb_rot_loss 
        self.use_acc_recon_loss = use_acc_recon_loss
        self.pred_last_frame = pred_last_frame
        self.use_vposer_loss = use_vposer_loss
        self.use_vel_loss = use_vel_loss

        #WheelPoser Modification
        self.upper_body_only = upper_body_only
        self.reduced_pose_output = reduced_pose_output
        self.mixed_imu_set = mixed_imu_set
        self.leave_one_out = leave_one_out
        self.video_data = video_data
        self.exp_setup = exp_setup
        self.upsample_copies = upsample_copies

        if device != None:
            if 'cpu' in device:
                self.device = torch.device(f'cpu')
            else:
                self.device = torch.device(f'cuda:{device}')
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.build_paths()

        self.loss_type = loss_type
    
    def build_paths(self):
        self.smpl_model_path = self.root_dir / "src/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"              # official SMPL model path
        self.raw_amass_path = self.root_dir / "src/data/dataset_raw/AMASS"
        self.raw_wheelposer_path = self.root_dir / "src/data/dataset_raw/WheelPoser"
        self.processed_amass_path = self.root_dir / "src/data/dataset_work/PROCESSED/AMASS"
        self.processed_wheelposer_path = self.root_dir / "src/data/dataset_work/PROCESSED/WheelPoser"
        self.combined_amass_path = self.root_dir / "src/data/dataset_work/COMBINED/AMASS"
        self.combined_wheelposer_path = self.root_dir / "src/data/dataset_work/COMBINED/WheelPoser"
        # self.processed_wheelposer_4 = self.root_dir / "src/data/dataset_work/4Joints/PROCESSED"
        # self.processed_amass_nn_ready_4 = self.root_dir / "src/data/dataset_work/4Joints/COMBINED/AMASS"        
        # self.processed_wheelposer_nn_ready_4 = self.root_dir / "src/data/dataset_work/4Joints/COMBINED/WheelPoser"
        # self.processed_wheelposer_leave_14_out = self.root_dir / "src/data/dataset_work/4Joints/COMBINED/Leave_14_out"
        # self.processed_wheelposer_leave_13_out = self.root_dir / "src/data/dataset_work/4Joints/COMBINED/Leave_13_out"

        # self.processed_wheelposer_wheelposer_am_nn_ready_4 = self.root_dir / "src/data/dataset_work/4Joints/COMBINED/WheelPoser_AM"
        # self.processed_wheelposer_mixed_nn_ready_4 = self.root_dir / "src/data/dataset_work/4Joints/COMBINED/WD_mixed"
        # self.processed_wheelposer_leave_one_out_wu = self.root_dir / "src/data/dataset_work/4Joints/COMBINED/WU_leave_one_out"
        # self.processed_wheelposer_video = self.root_dir / "src/data/dataset_work/4Joints/COMBINED/Video_data"
        # self.processed_wheelposer_category = self.root_dir / "src/data/dataset_work/4Joints/CATEGORY"

        if self.mkdir:
            if self.experiment != None:
                datestring = datetime.datetime.now().strftime("%m%d%Y-%H%M%S")
                if self.exp_setup != None:
                    self.checkpoint_path = self.root_dir / f"checkpoints/{self.experiment}-{self.model}-{self.exp_setup}-{datestring}"
                    self.checkpoint_path.mkdir(exist_ok=True, parents=True)
                else:
                    self.checkpoint_path = self.root_dir / f"checkpoints/{self.experiment}-{self.model}-{datestring}"
                    self.checkpoint_path.mkdir(exist_ok=True, parents=True)
            else:
                print("No experiment name give, can't create dir")

    max_sample_len = 150
    online_window = 26
    acc_scale = 30
    vel_scale = 3
    train_pct = 0.9
    batch_size = 256
    torch_seed = 0

limb2joints = {
    "LLeg": [1, 4, 7, 10],
    "RLeg": [2, 5, 8, 11],
    "LArm": [16, 18, 20, 22],
    "RArm": [17, 19, 21, 23],
    "Head": [15, 12],
    "Torso": [3, 6, 9, 13, 14]
}

limb2vertexkeys = {
    "LLeg": ["leftLeg", "leftToeBase", "leftFoot", "leftUpLeg"],
    "RLeg": ["rightUpLeg", "rightFoot", "rightLeg", "rightToeBase"],
    "LArm": ["leftArm", "leftHandIndex1", "leftForeArm", "leftHand", "leftShoulder"], 
    "RArm": ["rightArm", "rightHandIndex1", "rightForeArm", "rightHand", "rightShoulder"], 
    "Head": ["head", "neck"], 
    "Torso": ["spine1", "spine2", "spine", "hips"]
}

leaf_joints = [20, 21, 12] #WheelPoser Modification
leaf_joints_transpose = [20, 21, 7, 8, 12] 
