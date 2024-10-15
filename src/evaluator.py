r"""
    Basic evaluators, and evaluators that measure differences between poses/trans of MANO/SMPL/SMPLH model.
"""


__all__ = ['BinaryConfusionMatrixEvaluator', 'BinaryClassificationErrorEvaluator', 'PositionErrorEvaluator',
           'RotationErrorEvaluator', 'PerJointErrorEvaluator', 'MeanPerJointErrorEvaluator', 'MeshErrorEvaluator',
           'FullMotionEvaluator', 'IMUPoserEvaluator']


from src.articulate.math import *
import torch
from src.articulate.model import ParametricModel
from src.config import Config, limb2joints
from src.smpl import limb2vertices


class BasePoseEvaluator:
    r"""
    Base class for evaluators that evaluate motions.
    """
    def __init__(self, official_model_file: str, rep=RotationRepresentation.ROTATION_MATRIX, use_pose_blendshape=False,
            device=torch.device('cuda:0')):
        self.model = ParametricModel(official_model_file, use_pose_blendshape=use_pose_blendshape, device=device)
        self.rep = rep
        self.device = device

    def _preprocess(self, pose, shape=None, tran=None):
        pose = to_rotation_matrix(pose.to(self.device), self.rep).view(pose.shape[0], -1)
        shape = shape.to(self.device) if shape is not None else shape
        tran = tran.to(self.device) if tran is not None else tran
        return pose, shape, tran


class BinaryConfusionMatrixEvaluator:
    r"""
    Confusion matrix for binary classification tasks.

    The (i, j) entry stands for the number of instance i that is classified as j.
    """
    def __init__(self, is_after_sigmoid=False):
        r"""
        Init a binary confusion matrix evaluator.

        :param is_after_sigmoid: Whether a sigmoid function has been applied on the predicted values or not.
        """
        self.is_after_sigmoid = is_after_sigmoid

    def __call__(self, p: torch.Tensor, t: torch.Tensor):
        r"""
        Get the confusion matrix.

        :param p: Predicted values (0 ~ 1 if is_after_sigmoid is True) in shape [*].
        :param t: True values (0 or 1) in shape [*].
        :return: Confusion matrix in shape [2, 2].
        """
        positive, negative = 0, 1
        p = (p > 0.5).float() if self.is_after_sigmoid else (p > 0).float()
        tp = ((p == positive) & (t == positive)).sum()
        fn = ((p == negative) & (t == positive)).sum()
        fp = ((p == positive) & (t == negative)).sum()
        tn = ((p == negative) & (t == negative)).sum()
        return torch.tensor([[tp, fn], [fp, tn]])


class BinaryClassificationErrorEvaluator(BinaryConfusionMatrixEvaluator):
    r"""
    Precision, recall, and f1 score for both positive and negative samples for binary classification tasks.
    """
    def __init__(self, is_after_sigmoid=False):
        r"""
        Init a binary classification error evaluator.

        :param is_after_sigmoid: Whether a sigmoid function has been applied on the predicted values or not.
        """
        super(BinaryClassificationErrorEvaluator, self).__init__(is_after_sigmoid)

    def __call__(self, p: torch.Tensor, t: torch.Tensor):
        r"""
        Get the precision, recall, and f1 score for both positive and negative samples.

        :param p: Predicted values (0 ~ 1 if is_after_sigmoid is True) in shape [*].
        :param t: True values (0 or 1) in shape [*].
        :return: Tensor in shape [3, 2] where column 0 and 1 are the precision, recall, and f1 score
                 for positive(0) and negative(1) samples respectively.
        """
        tp, fn, fp, tn = super(BinaryClassificationErrorEvaluator, self).__call__(p, t).view(-1)

        precision_positive = tp.float() / (tp + fp)
        recall_positive = tp.float() / (tp + fn)
        f1_positive = 2 / (1 / precision_positive + 1 / recall_positive)

        precision_negative = tn.float() / (tn + fn)
        recall_negative = tn.float() / (tn + fp)
        f1_negative = 2 / (1 / precision_negative + 1 / recall_negative)

        return torch.tensor([[precision_positive, precision_negative],
                             [recall_positive, recall_negative],
                             [f1_positive, f1_negative]])


class PositionErrorEvaluator:
    r"""
    Mean distance between two sets of points. Distances are defined as vector p-norm.
    """
    def __init__(self, dimension=3, p=2):
        r"""
        Init a distance error evaluator.

        Notes
        -----
        The two tensors being evaluated will be reshape to [n, dimension] and be regarded as n points.
        Then the average of p-norms of the difference of all corresponding points will be returned.

        Args
        -----
        :param dimension: Dimension of the vector space. By default 3.
        :param p: Distance will be evaluated by vector p-norm. By default 2.
        """
        self.dimension = dimension
        self.p = p

    def __call__(self, p: torch.Tensor, t: torch.Tensor):
        r"""
        Get the mean p-norm distance between two sets of points.

        :param p: Tensor that can reshape to [n, dimension] that stands for n points.
        :param t: Tensor that can reshape to [n, dimension] that stands for n points.
        :return: Mean p-norm distance between all corresponding points.
        """
        return (p.view(-1, self.dimension) - t.view(-1, self.dimension)).norm(p=self.p, dim=1).mean()


class RotationErrorEvaluator:
    r"""
    Mean angle between two sets of rotations. Angles are in degrees.
    """
    def __init__(self, rep=RotationRepresentation.ROTATION_MATRIX):
        r"""
        Init a rotation error evaluator.

        :param rep: The rotation representation used in the input.
        """
        self.rep = rep

    def __call__(self, p: torch.Tensor, t: torch.Tensor):
        r"""
        Get the mean angle between to sets of rotations.

        :param p: Tensor that can reshape to [n, rep_dim] that stands for n rotations.
        :param t: Tensor that can reshape to [n, rep_dim] that stands for n rotations.
        :return: Mean angle in degrees between all corresponding rotations.
        """
        return radian_to_degree(angle_between(p, t, self.rep).mean())


class PerJointErrorEvaluator(BasePoseEvaluator):
    r"""
    Position and local/global rotation error of each joint.
    """
    def __init__(self, official_model_file: str, align_joint=None, rep=RotationRepresentation.ROTATION_MATRIX,
                 device=torch.device('cpu')):
        r"""
        Init a PJE Evaluator.

        :param official_model_file: Path to the official SMPL/MANO/SMPLH model to be loaded.
        :param align_joint: Which joint to align. (e.g. SMPLJoint.ROOT). By default the root.
        :param rep: The rotation representation used in the input poses.
        :param device: torch.device, cpu or cuda.
        """
        super().__init__(official_model_file, rep, device=device)
        self.align_joint = 0 if align_joint is None else align_joint.value

    def __call__(self, pose_p: torch.Tensor, pose_t: torch.Tensor):
        r"""
        Get position and local/global rotation errors of all joints.

        :param pose_p: Predicted pose or the first pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :param pose_t: True pose or the second pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :return: Tensor in shape [3, num_joint] where the ith column is the position error,
                 local rotation error, and global rotation error (in degrees) of the ith joint.
        """
        batch_size = pose_p.shape[0]
        pose_local_p, _, _ = self._preprocess(pose_p)
        pose_local_t, _, _ = self._preprocess(pose_t)
        pose_global_p, joint_p = self.model.forward_kinematics(pose_local_p)
        pose_global_t, joint_t = self.model.forward_kinematics(pose_local_t)
        offset_from_p_to_t = (joint_t[:, self.align_joint] - joint_p[:, self.align_joint]).unsqueeze(1)
        joint_p = joint_p + offset_from_p_to_t
        position_error_array = (joint_p - joint_t).norm(dim=2).mean(dim=0)
        local_rotation_error_array = angle_between(pose_local_p, pose_local_t).view(batch_size, -1).mean(dim=0)
        global_rotation_error_array = angle_between(pose_global_p, pose_global_t).view(batch_size, -1).mean(dim=0)
        return torch.stack((position_error_array,
                            radian_to_degree(local_rotation_error_array),
                            radian_to_degree(global_rotation_error_array)))


class MeanPerJointErrorEvaluator(PerJointErrorEvaluator):
    r"""
    Mean position and local/global rotation error of all joints.
    """
    def __init__(self, official_model_file: str, align_joint=None, rep=RotationRepresentation.ROTATION_MATRIX,
                 device=torch.device('cpu')):
        r"""
        Init a MPJE Evaluator.

        :param official_model_file: Path to the official SMPL/MANO/SMPLH model to be loaded.
        :param align_joint: Which joint to align. (e.g. SMPLJoint.ROOT). By default the root.
        :param rep: The rotation representation used in the input poses.
        :param device: torch.device, cpu or cuda.
        """
        super().__init__(official_model_file, align_joint, rep, device)

    def __call__(self, pose_p: torch.Tensor, pose_t: torch.Tensor):
        r"""
        Get mean position and local/global rotation errors of all joints.

        :param pose_p: Predicted pose or the first pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :param pose_t: True pose or the second pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :return: Tensor in shape [3] containing the mean position error,
                 local rotation error, and global rotation error (in degrees) of all joints.
        """
        error_array = super(MeanPerJointErrorEvaluator, self).__call__(pose_p, pose_t)
        return error_array.mean(dim=1)


class MeshErrorEvaluator(BasePoseEvaluator):
    r"""
    Mean mesh vertex position error.
    """
    def __init__(self, official_model_file: str, align_joint=None, rep=RotationRepresentation.ROTATION_MATRIX,
                 use_pose_blendshape=False, device=torch.device('cpu')):
        r"""
        Init a mesh error evaluator.

        :param official_model_file: Path to the official SMPL/MANO/SMPLH model to be loaded.
        :param align_joint: Which joint to align. (e.g. SMPLJoint.ROOT). By default the root.
        :param rep: The rotation representation used in the input poses.
        :param use_pose_blendshape: Whether to use pose blendshape or not.
        :param device: torch.device, cpu or cuda.
        """
        super().__init__(official_model_file, rep, use_pose_blendshape, device=device)
        self.align_joint = 0 if align_joint is None else align_joint.value

    def __call__(self, pose_p: torch.Tensor, pose_t: torch.Tensor,
                 shape_p: torch.Tensor = None, shape_t: torch.Tensor = None):
        r"""
        Get mesh vertex position error.

        :param pose_p: Predicted pose or the first pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :param pose_t: True pose or the second pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :param shape_p: Predicted shape that can expand to [batch_size, 10]. Use None for the mean(zero) shape.
        :param shape_t: True shape that can expand [batch_size, 10]. Use None for the mean(zero) shape.
        :return: Mean mesh vertex position error.
        """
        pose_p, shape_p, _ = self._preprocess(pose_p, shape_p)
        pose_t, shape_t, _ = self._preprocess(pose_t, shape_t)
        _, joint_p, mesh_p = self.model.forward_kinematics(pose_p, shape_p, calc_mesh=True)
        _, joint_t, mesh_t = self.model.forward_kinematics(pose_t, shape_t, calc_mesh=True)
        offset_from_p_to_t = (joint_t[:, self.align_joint] - joint_p[:, self.align_joint]).unsqueeze(1)
        mesh_error = (mesh_p + offset_from_p_to_t - mesh_t).norm(dim=2).mean()
        return mesh_error


class FullMotionEvaluator(BasePoseEvaluator):
    r"""
    Evaluator for full motions (pose sequences with global translations). Plenty of metrics.
    """
    def __init__(self, official_model_file: str, align_joint=None, rep=RotationRepresentation.ROTATION_MATRIX,
                 use_pose_blendshape=False, fps=60, joint_mask=None, device=torch.device('cpu')):
        r"""
        Init a full motion evaluator.

        :param official_model_file: Path to the official SMPL/MANO/SMPLH model to be loaded.
        :param align_joint: Which joint to align. (e.g. SMPLJoint.ROOT). By default the root.
        :param rep: The rotation representation used in the input poses.
        :param use_pose_blendshape: Whether to use pose blendshape or not.
        :param joint_mask: If not None, local angle error, global angle error, and joint position error
                           for these joints will be calculated additionally.
        :param fps: Motion fps, by default 60.
        :param device: torch.device, cpu or cuda.
        """
        super(FullMotionEvaluator, self).__init__(official_model_file, rep, use_pose_blendshape, device=device)
        self.align_joint = 0 if align_joint is None else align_joint.value
        self.fps = fps
        self.joint_mask = joint_mask

    def __call__(self, pose_p, pose_t, shape_p=None, shape_t=None, tran_p=None, tran_t=None):
        r"""
        Get the measured errors. The returned tensor in shape [11, 2] contains mean and std of:
          0.  Joint position error (align_joint position aligned).
          1.  Vertex position error (align_joint position aligned).
          2.  Joint local angle error (in degrees).
          3.  Joint global angle error (in degrees).
          4.  Predicted motion jerk (with global translation).
          5.  True motion jerk (with global translation).
          6.  Translation error (mean root translation error per second, using a time window size of 1s).
          7.  Masked joint position error (align_joint position aligned, zero if mask is None).
          8.  Masked joint local angle error. (in degrees, zero if mask is None).
          9.  Masked joint global angle error. (in degrees, zero if mask is None).
          10. Masked vertex position error. (align_joint position aligned, zero if mask is None).
          11. Tracking error. Mean joint error of the whole motions (including global translation and rotation).
          12. Masker predicted motion jerk (with global translation, zero if mask is None).
          13. Wrist position error (both left and right)
          14. elbow position error (both left and right)
          15. wrist travel distance error (both left and right)
          16. elbow travel distance error (both left and right)






        :param pose_p: Predicted pose or the first pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :param pose_t: True pose or the second pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :param shape_p: Predicted shape that can expand to [batch_size, 10]. Use None for the mean(zero) shape.
        :param shape_t: True shape that can expand to [batch_size, 10]. Use None for the mean(zero) shape.
        :param tran_p: Predicted translations in shape [batch_size, 3]. Use None for zeros.
        :param tran_t: True translations in shape [batch_size, 3]. Use None for zeros.
        :return: Tensor in shape [11, 2] for the mean and std of all errors.
        """
        f = self.fps
        pose_local_p, shape_p, tran_p = self._preprocess(pose_p, shape_p, tran_p)
        pose_local_t, shape_t, tran_t = self._preprocess(pose_t, shape_t, tran_t)
        pose_global_p, joint_p, vertex_p = self.model.forward_kinematics(pose_local_p, shape_p, tran_p, calc_mesh=True)
        pose_global_t, joint_t, vertex_t = self.model.forward_kinematics(pose_local_t, shape_t, tran_t, calc_mesh=True)

        offset_from_p_to_t = (joint_t[:, self.align_joint] - joint_p[:, self.align_joint]).unsqueeze(1)
        tre = (joint_p - joint_t).norm(dim=2)                         # N, J
        ve = (vertex_p + offset_from_p_to_t - vertex_t).norm(dim=2)   # N, J
        je = (joint_p + offset_from_p_to_t - joint_t).norm(dim=2)     # N, J
        lae = radian_to_degree(angle_between(pose_local_p, pose_local_t).view(pose_p.shape[0], -1))           # N, J
        gae = radian_to_degree(angle_between(pose_global_p, pose_global_t).view(pose_p.shape[0], -1))         # N, J
        jkp = ((joint_p[3:] - 3 * joint_p[2:-1] + 3 * joint_p[1:-2] - joint_p[:-3]) * (f ** 3)).norm(dim=2)   # N, J
        jkt = ((joint_t[3:] - 3 * joint_t[2:-1] + 3 * joint_t[1:-2] - joint_t[:-3]) * (f ** 3)).norm(dim=2)   # N, J
        te = ((joint_p[f:, :1] - joint_p[:-f, :1]) - (joint_t[f:, :1] - joint_t[:-f, :1])).norm(dim=2)        # N, 1
        mje = je[:, self.joint_mask] if self.joint_mask is not None else torch.zeros(1)     # N, mJ
        mlae = lae[:, self.joint_mask] if self.joint_mask is not None else torch.zeros(1)   # N, mJ
        mgae = gae[:, self.joint_mask] if self.joint_mask is not None else torch.zeros(1)   # N, mJ
        mve = ve[:, self.joint_mask] if self.joint_mask is not None else torch.zeros(1)     # N, mJ
        mjkp = jkp[:, self.joint_mask] if self.joint_mask is not None else torch.zeros(1)   # N, mJ

        corrected_joint_p = joint_p + offset_from_p_to_t

        wrist_p = corrected_joint_p[:, [20, 21]]
        elbow_p = corrected_joint_p[:, [18, 19]]
        wrist_t = joint_t[:, [20, 21]]
        elbow_t = joint_t[:, [18, 19]]

        #travel distance of wrist
        wrist_travel_distance_p = (wrist_p[1:] - wrist_p[:-1]).norm(dim=2).sum(dim=0).sum()
        wrist_travel_distance_t = (wrist_t[1:] - wrist_t[:-1]).norm(dim=2).sum(dim=0).sum()
        # absolute_distance_percentage_error_wrist = abs(wrist_travel_distance_p - wrist_travel_distance_t) / wrist_travel_distance_t
        absolute_distance_error_wrist = abs(wrist_travel_distance_p - wrist_travel_distance_t)

        #travel distance of elbow
        elbow_travel_distance_p = (elbow_p[1:] - elbow_p[:-1]).norm(dim=2).sum(dim=0).sum()
        elbow_travel_distance_t = (elbow_t[1:] - elbow_t[:-1]).norm(dim=2).sum(dim=0).sum()
        # absolute_distance_percentage_error_elbow = abs(elbow_travel_distance_p - elbow_travel_distance_t) / elbow_travel_distance_t
        absolute_distance_error_elbow = abs(elbow_travel_distance_p - elbow_travel_distance_t)



        wrist_je = je[:, [20, 21]]
        elbow_je = je[:, [18, 19]]



        return torch.tensor([[je.mean(),   je.std(dim=0).mean()],
                             [ve.mean(),   ve.std(dim=0).mean()],
                             [lae.mean(),  lae.std(dim=0).mean()],
                             [gae.mean(),  gae.std(dim=0).mean()],
                             [jkp.mean(),  jkp.std(dim=0).mean()],
                             [jkt.mean(),  jkt.std(dim=0).mean()],
                             [te.mean(),   te.std(dim=0).mean()],
                             [mje.mean(),  mje.std(dim=0).mean()],
                             [mlae.mean(), mlae.std(dim=0).mean()],
                             [mgae.mean(), mgae.std(dim=0).mean()],
                             [mve.mean(),  mve.std(dim=0).mean()],
                             [tre.mean(),  tre.std(dim=0).mean()],
                             [mjkp.mean(), mjkp.std(dim=0).mean()],
                             [wrist_je.mean(), wrist_je.std(dim=0).mean()],
                             [elbow_je.mean(), elbow_je.std(dim=0).mean()],
                             [absolute_distance_error_wrist, 0],
                             [absolute_distance_error_elbow, 0],
                             [wrist_travel_distance_t, 0],
                             [elbow_travel_distance_t, 0]])

class ReducedPoseEvaluator:
    names = ['SIP Error (deg)', 'Angle Error (deg)', 'Joint Error (cm)', 'Vertex Error (cm)', 'Jitter Error (100m/s^3)']

    def __init__(self, config):
        self._base_motion_loss_fn = FullMotionEvaluator(config.smpl_model_path, joint_mask=torch.tensor([1, 2, 16, 17]), device=config.device)
        self.ignored_joint_mask = torch.tensor([0, 7, 8, 10, 11, 20, 21, 22, 23])

    def __call__(self, pose_p, pose_t):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        pose_p[:, self.ignored_joint_mask] = torch.eye(3, device=pose_p.device)
        pose_t[:, self.ignored_joint_mask] = torch.eye(3, device=pose_t.device)
        errs = self._base_motion_loss_fn(pose_p=pose_p, pose_t=pose_t)
        return torch.stack([errs[9], errs[3], errs[0] * 100, errs[1] * 100, errs[4] / 100])
    

class WheelPoserEvaluator:
    names = ['Angle Error (deg)', 'Joint Error (cm)', 'Vertex Error (cm)', 'Jitter Error (100m/s^3)', 'Wrist Error (cm)', 'Elbow Error (cm)', 'Wrist Travel Error (m)', 'Elbow Travel Error (m)', 'Wrist Travel Distance GT (m)', 'Elbow Travel Distance GT (m)']

    def __init__(self, config):
        self._base_motion_loss_fn = FullMotionEvaluator(config.smpl_model_path, joint_mask=torch.tensor([0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]), device=config.device)
        self.ignored_joint_mask = torch.tensor([0, 7, 8, 10, 11, 20, 21, 22, 23])
        # self.ignored_joint_mask = torch.tensor([0, 13, 14, 6, 9, 7, 8, 10, 11, 20, 21, 22, 23])
        
        # # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        # # Pelvis, spine1, spine2, spine3, neck, l_collar, r_collar, head, l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist, l_hand, r_hand
        # self.ignored_joint_mask = torch.tensor([0, 12, 13, 14, 15])
        
    def __call__(self, pose_p, pose_t):
        pose_p = pose_p.clone().view(-1, 16, 3, 3)
        pose_t = pose_t.clone().view(-1, 16, 3, 3)
        # add to full body pose
        #TODO
        lower_body_pose_p = torch.eye(3, device=pose_p.device).view(1, 3, 3).unsqueeze(0).repeat(pose_p.shape[0], 1, 1, 1)
        lower_body_pose_t = torch.eye(3, device=pose_p.device).view(1, 3, 3).unsqueeze(0).repeat(pose_t.shape[0], 1, 1, 1)
        pose_p = torch.cat([pose_p[:,:1, :, :], lower_body_pose_p, lower_body_pose_p, pose_p[:,1:2, :, :], lower_body_pose_p, lower_body_pose_p, pose_p[:,2:3, :, :], lower_body_pose_p, lower_body_pose_p,
                            pose_p[:,3:4, :, :], lower_body_pose_p, lower_body_pose_p, pose_p[:,4:, :, :]], dim=1)

        pose_t = torch.cat([pose_t[:,:1, :, :], lower_body_pose_t, lower_body_pose_t, pose_t[:,1:2, :, :], lower_body_pose_t, lower_body_pose_t, pose_t[:,2:3, :, :], lower_body_pose_t, lower_body_pose_t,
                            pose_t[:,3:4, :, :], lower_body_pose_t, lower_body_pose_t, pose_t[:,4:, :, :]], dim=1)
        
        pose_p[:, self.ignored_joint_mask] = torch.eye(3, device=pose_p.device)
        pose_t[:, self.ignored_joint_mask] = torch.eye(3, device=pose_t.device)
        errs = self._base_motion_loss_fn(pose_p=pose_p, pose_t=pose_t)
        return torch.stack([errs[9], errs[7] * 100, errs[10] * 100, errs[12] / 100, errs[13]* 100, errs[14]* 100, errs[15], errs[16], errs[17], errs[18]])


def get_limb_metrics(je, ve, lae, gae, jkp, jkt, limb):
    limb_je = je[:, limb2joints[limb]]
    limb_ve = ve[:, limb2vertices(limb)]
    limb_lae = lae[:, limb2joints[limb]]
    limb_gae = gae[:, limb2joints[limb]]
    limb_jkp = jkp[:, limb2joints[limb]]
    limb_jkt = jkt[:, limb2joints[limb]]
    
    return torch.Tensor([[limb_je.mean(), limb_je.std(dim=0).mean()],
                        [limb_ve.mean(), limb_ve.std(dim=0).mean()],
                        [limb_lae.mean(), limb_lae.std(dim=0).mean()],
                        [limb_gae.mean(), limb_gae.std(dim=0).mean()],
                        [limb_jkp.mean(), limb_jkp.std(dim=0).mean()],
                        [limb_jkt.mean(), limb_jkt.std(dim=0).mean()]])
