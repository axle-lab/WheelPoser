import torch
import numpy as np
import pybullet as p
import src.articulate as art
from src.articulate.utils.bullet import *
from src.articulate.utils.rbdl import *
from src.utils import *
from qpsolvers import solve_qp
from src.config import paths



class PhysicsOptimizer:
    test_contact_joints = ['LHIP', 'RHIP', 'SPINE1', 'LKNEE', 'RKNEE', 'SPINE2',
                           'SPINE3', 'LSHOULDER', 'RSHOULDER', 'HEAD',
                           'LELBOW', 'RELBOW', 'LHAND', 'RHAND', 'LFOOT', 'RFOOT'
                           ]  # 'LANKLE', 'RANKLE', 'NECK', 'LWRIST', 'RWRIST', 'LCLAVICLE', 'RCLAVICLE'
    fixed_contact_joints = ['LHIP', 'RHIP', 'LFOOT', 'RFOOT']
    seat_height = -0.085
    footrest_height = -0.4


    def __init__(self, debug=False):
        mu = 0.6
        supp_poly_size = 0.2
        self.debug = debug
        self.model = RBDLModel(paths.physics_model_file, update_kinematics_by_hand=True)
        self.params = read_debug_param_values_from_json(paths.physics_parameter_file)
        self.friction_constraint_matrix = np.array([[np.sqrt(2), -mu, 0],
                                                    [-np.sqrt(2), -mu, 0],
                                                    [0, -mu, np.sqrt(2)],
                                                    [0, -mu, -np.sqrt(2)]])

        self.support_polygon = np.array([[-supp_poly_size / 2,  0,  -supp_poly_size / 2],
                                         [ supp_poly_size / 2,  0,  -supp_poly_size / 2],
                                         [-supp_poly_size / 2,  0,   supp_poly_size / 2],
                                         [ supp_poly_size / 2,  0,   supp_poly_size / 2]])
        if debug:
            p.connect(p.GUI)
            p.configureDebugVisualizer(flag=p.COV_ENABLE_Y_AXIS_UP, enable=1)
            self.id_robot = p.loadURDF(paths.physics_model_file, [0, 0, 0], useFixedBase=False, flags=p.URDF_MERGE_FIXED_LINKS)
            change_color(self.id_robot, [198 / 255, 238 / 255, 0, 1.0])
            p.loadURDF(paths.plane_file, [0, -0.881, 0.0], [-0.7071068, 0, 0, 0.7071068])
            load_debug_params_into_bullet_from_json(paths.physics_parameter_file)


        # states
        self.last_x = []
        self.q = None
        self.qdot = np.zeros(self.model.qdot_size)
        self.reset_states()

    def reset_states(self):
        self.last_x = []
        self.q = None
        self.qdot = np.zeros(self.model.qdot_size)


    
    def optimize_frame(self, pose, contact=None):
        q_ref = smpl_to_rbdl(pose, torch.zeros(3))[0]
        # c_ref = contact.sigmoid().numpy()
        q = self.q
        qdot = self.qdot


        # if q is None:
        #     self.q = q_ref
        #     return pose

        if q is None:
            self.q = q_ref
            return pose, torch.zeros(75)
        
        #determine the contact joints and points
        self.model.update_kinematics(q, qdot, np.zeros(self.model.qdot_size))
        Js = [np.empty((0, self.model.qdot_size))]
        collision_points, collision_joints = [], []
        for joint_name in self.fixed_contact_joints:
            joint_id = vars(Body)[joint_name]
            pos = self.model.calc_body_position(q, joint_id)
            collision_joints.append(joint_name)
            # if joint_id == Body.LHIP or joint_id == Body.RHIP:
            #     # print("hip", pos[1])
            #     pos[1] = self.seat_height - pos[1]
            # else:
            #     # print("foot", pos[1])
            #     pos[1] = self.footrest_height - pos[1]
            for ps in self.support_polygon + pos:
                collision_points.append(ps)
                pb = self.model.calc_base_to_body_coordinates(q, joint_id, ps)
                Js.append(self.model.calc_point_Jacobian(q, joint_id, pb))
        Js = np.vstack(Js)
        nc = len(collision_points)


        # minimize   ||A1 * qddot - b1||^2     for A1, b1 in zip(As1, bs1)
        #            + ||A2 * lambda - b2||^2  for A2, b2 in zip(As2, bs2)
        #            + ||A3 * tau - b3||^2     for A3, b3 in zip(As3, bs3)
        # s.t.       G1 * qddot <= h1          for G1, h1 in zip(Gs1, hs1)
        #            G2 * lambda <= h2         for G2, h2 in zip(Gs2, hs2)
        #            G3 * tau <= h3            for G3, h3 in zip(Gs3, hs3)
        #            A_ * x = b_
        As1, bs1, As2, bs2, As3, bs3 = [np.zeros((0, self.model.qdot_size))], [np.empty(0)], [np.empty((0, nc * 3))], \
                                       [np.empty(0)], [np.zeros((0, self.model.qdot_size))], [np.empty(0)]
        Gs1, hs1, Gs2, hs2, Gs3, hs3 = [np.zeros((0, self.model.qdot_size))], [np.empty(0)], [np.empty((0, nc * 3))], \
                                       [np.empty(0)], [np.zeros((0, self.model.qdot_size))], [np.empty(0)]
        A_, b_ = None, None


        # joint angle PD controller
        if True:
            A = np.hstack((np.zeros((self.model.qdot_size - 3, 3)), np.eye((self.model.qdot_size - 3))))
            b = self.params['kp_angular'] * art.math.angle_difference(q_ref[3:], q[3:]) - self.params['kd_angular'] * qdot[3:]
            As1.append(A)  # 72 * 75
            bs1.append(b)  # 72


        # lambda size
        if False:
            As2.append(np.eye(nc * 3) * self.params['coeff_lambda_old'])
            bs2.append(np.zeros(nc * 3))

        # Signorini’s conditions of lambda
        # Modify to wheelchair seat's and footrest's position
        # if True:
        #     if nc != 0:
        #         A = [np.eye(3) * max(cp[1] - self.params['floor_y'], 0.005) for cp in collision_points]
        #         A = art.math.block_diagonal_matrix_np(A)
        #         As2.append(A * self.params['coeff_lambda']) #48*48
        #         bs2.append(np.zeros(nc * 3)) #48,
        if True:
            if nc != 0:
                # A = [np.eye(3) * max(cp[1], 0.005) for cp in collision_points]
                A = [np.eye(3) * 0.005 for cp in collision_points]
                A = art.math.block_diagonal_matrix_np(A)
                As2.append(A * self.params['coeff_lambda']) #48*48
                bs2.append(np.zeros(nc * 3)) #48,

        # tau size
        if True:
            As3.append(art.math.block_diagonal_matrix_np([
                np.eye(6) * self.params['coeff_virtual'],
                np.eye(self.model.qdot_size - 6) * self.params['coeff_tau']
            ])) #72*75
            bs3.append(np.zeros(self.model.qdot_size)) #75,

        # contacting body joint velocity
        if True:
            for joint_name in self.fixed_contact_joints:
                joint_id = vars(Body)[joint_name]
                pos = self.model.calc_body_position(q, joint_id)
                J = self.model.calc_point_Jacobian(q, joint_id)
                v = self.model.calc_point_velocity(q, qdot, joint_id)
                Gs1.append(-self.params['delta_t'] * J)
                hs1.append(v - [-1e-1, 0, -1e-1])
                Gs1.append(self.params['delta_t'] * J)
                hs1.append(-v + [1e-1, 1e2, 1e-1])



        # contacting point velocity
        if False:
            for joint_name in self.fixed_contact_joints:
                joint_id = vars(Body)[joint_name]
                J = self.model.calc_point_Jacobian(q, joint_id)
                v = self.model.calc_point_velocity(q, qdot, joint_id)
                th = -np.log(0.84999 / 0.85)
                th_y = (self.params['floor_y'] - pos[1]) / self.params['delta_t']
                Gs1.append(-self.params['delta_t'] * J)
                hs1.append(v - [-th, th_y, -th])
                Gs1.append(self.params['delta_t'] * J)
                hs1.append(-v + [th, max(th, th_y) + 1e-6, th])

        # GRF friction cone constraint
        if True:
            if nc > 0:
                Gs2.append(art.math.block_diagonal_matrix_np([self.friction_constraint_matrix] * nc))
                hs2.append(np.zeros(nc * 4))

        # equation of motion (equality constraint)
        if True:
            M = self.model.calc_M(q)
            h = self.model.calc_h(q, qdot)
            A_ = np.hstack((-M, Js.T, np.eye(self.model.qdot_size)))
            b_ = h



        As1, bs1, As2, bs2, As3, bs3 = np.vstack(As1), np.concatenate(bs1), np.vstack(As2), np.concatenate(bs2), np.vstack(As3), np.concatenate(bs3)
        Gs1, hs1, Gs2, hs2, Gs3, hs3 = np.vstack(Gs1), np.concatenate(hs1), np.vstack(Gs2), np.concatenate(hs2), np.vstack(Gs3), np.concatenate(hs3)
        G_ = art.math.block_diagonal_matrix_np([Gs1, Gs2, Gs3])
        h_ = np.concatenate((hs1, hs2, hs3))
        P_ = art.math.block_diagonal_matrix_np([np.dot(As1.T, As1), np.dot(As2.T, As2), np.dot(As3.T, As3)])
        q_ = np.concatenate((-np.dot(As1.T, bs1), -np.dot(As2.T, bs2), -np.dot(As3.T, bs3)))

        # fast solvers are less accurate/robust, and may fail
        init = self.last_x if len(self.last_x) == len(q_) else None
        # x = solve_qp(P_, q_, G_, h_, A_, b_, solver='quadprog', initvals=init)
        x = solve_qp(P_, q_, G_, h_, A_, b_, solver='cvxopt', initvals=init)

        if x is None or np.linalg.norm(x) > 10000:
            x = solve_qp(P_, q_, G_, h_, A_, b_, solver='cvxopt', initvals=init)

        qddot = x[:self.model.qdot_size]
        GRF = x[self.model.qdot_size:-self.model.qdot_size]
        tau = x[-self.model.qdot_size:]

        qdot = qdot + qddot * self.params['delta_t']
        q = q + qdot * self.params['delta_t']
        self.q = q
        self.qdot = qdot
        self.last_x = x



        if self.debug:
            # self.clock.tick(60)   # please install pygame
            set_pose(self.id_robot, q)
            self.params = read_debug_param_values_from_bullet()

            if False:   # visualize GRF (no smoothing)
                p.removeAllUserDebugItems()
                for point, force in zip(collision_points, GRF.reshape(-1, 3)):
                    p.addUserDebugLine(point, point + force * 1e-2, [1, 0, 0])

        pose_opt, tran_opt = rbdl_to_smpl(q)
        pose_opt = torch.from_numpy(pose_opt).float()[0]
        tran_opt = torch.from_numpy(tran_opt).float()[0]
        return pose_opt, torch.from_numpy(tau)
        # return pose_opt








