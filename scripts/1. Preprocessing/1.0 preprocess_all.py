r"""
    Preprocess AMASS and DIP dataset
"""

import torch
import os
import numpy as np
from tqdm import tqdm
from src import articulate as art
from src.config import paths, amass_datasets_updated
import glob
from src.config import Config
import pickle

config = Config(project_root_dir="./")

def process_amass(smooth_n = 4):
    def _syn_acc(v):
        r"""
        Synthesize accelerations from vertex positions
        """
        mid = smooth_n // 2
        acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
        acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
        if mid != 0:
            acc[smooth_n:-smooth_n] = torch.stack(
                [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
                 for i in range(0, v.shape[0] - smooth_n * 2)])
        return acc
    

    # vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])
    # ji_mask = torch.tensor([18, 19, 4, 5, 15, 0])

    # WheelPoser left elbow, right elbow, head, pelvis
    vi_mask = torch.tensor([1961, 5424, 411, 3021])
    ji_mask = torch.tensor([18, 19, 15, 0])

    body_model = art.ParametricModel(paths.smpl_file)

    try:
        processed = [fpath.name for fpath in (config.processed_amass_path).iterdir()]
    except:
        processed = []

    for ds_name in amass_datasets_updated:
        print('\Reading', ds_name)
        if ds_name in processed:
            print('\Already processed', ds_name)
            continue
        data_pose, data_trans, data_beta, length = [], [], [], []
        print('\Processing', ds_name)

        #Default AMASS
        for npz_name in tqdm(glob.glob(os.path.join(paths.raw_amass_dir, ds_name, '*/*_poses.npz'))):

        #for GRAB and SOMA use this
        # for npz_name in tqdm(glob.glob(os.path.join(paths.raw_amass_dir, ds_name, '*/*_stageii.npz'))):
            # print(npz_name)
            try: cdata = np.load(npz_name)
            except: continue

            # framerate = int(cdata['mocap_framerate'])
            try: framerate = int(cdata['mocap_framerate'])
            except: continue

            if framerate ==120: step = 2
            elif framerate == 60 or framerate == 59: step = 1
            else: continue

            data_pose.extend(cdata['poses'][::step].astype(np.float32))
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            length.append(cdata['poses'][::step].shape[0])
    
        if len(data_pose) == 0:
            print(f"AMASS dataset, {ds_name} not supported")
            continue

        length = torch.tensor(length, dtype=torch.int)
        shape = torch.tensor(np.asarray(data_beta, np.float32))
        tran = torch.tensor(np.asarray(data_trans, np.float32))
        pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
        pose[:, 23] = pose[:, 37]     # right hand
        pose = pose[:, :24].clone()   # only use body

        # align AMASS global fame with DIP
        amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
        tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
        pose[:, 0] = art.math.rotation_matrix_to_axis_angle(
            amass_rot.matmul(art.math.axis_angle_to_rotation_matrix(pose[:, 0])))
        

        # go through the pose to filter out flipping motions
        b = 0
        filtered_pose = []
        filtered_tran = []
        filtered_shape = []
        filtered_length = []
        for i, l in tqdm(list(enumerate(length))):
            p = pose[b:b + l].view(-1, 24, 3)
            current_shape = shape[i]
            new_length = 0
            up_vector = torch.tensor([0, 1, 0], dtype=torch.float32)
            grot, joint, vert = body_model.forward_kinematics(art.math.axis_angle_to_rotation_matrix(p).view(-1, 24, 3, 3), shape[i], tran[b:b + l], calc_mesh=True)
            spine_y_axis = grot[:, 3, 1]
            pelvis_x_axis = grot[:, 0, 0]
            pelvis_z_axis = grot[:, 0, 2]

            #angle between up vector and spine up vector
            angle_spine_y = torch.acos(torch.sum(spine_y_axis * up_vector, dim=1) / (torch.norm(spine_y_axis, dim=1) * torch.norm(up_vector, dim=0)))
            angle_spine_y = torch.rad2deg(angle_spine_y)

            angle_pelvis_x = torch.acos(torch.sum(pelvis_x_axis * up_vector, dim=1) / (torch.norm(pelvis_x_axis, dim=1) * torch.norm(up_vector, dim=0)))
            angle_pelvis_x = torch.rad2deg(angle_pelvis_x)

            angle_pelvis_z = torch.acos(torch.sum(pelvis_z_axis * up_vector, dim=1) / (torch.norm(pelvis_z_axis, dim=1) * torch.norm(up_vector, dim=0)))
            angle_pelvis_z = torch.rad2deg(angle_pelvis_z)

            for j in range(p.shape[0] - 1):
                if angle_spine_y[j] < 140 and angle_pelvis_x[j] < 145 and angle_pelvis_x[j] > 35 and angle_pelvis_z[j] < 135 and angle_pelvis_z[j] > 45:
                    filtered_pose.append(p[j].clone())
                    filtered_tran.append(tran[b + j].clone())
                    new_length += 1
                else:
                    if new_length > 0:
                        filtered_shape.append(current_shape)
                        filtered_length.append(new_length)
                        new_length = 0
            if new_length > 0:
                filtered_shape.append(current_shape)
                filtered_length.append(new_length)
            b += l

        filtered_pose = torch.stack(filtered_pose)
        filtered_tran = torch.stack(filtered_tran)
        #turn list of tensor to tensor
        filtered_shape = torch.stack(filtered_shape, dim=0)
        filtered_length = torch.tensor(filtered_length, dtype=torch.int)

        length = filtered_length
        shape = filtered_shape
        tran = filtered_tran
        pose = filtered_pose

        print('Synthesizing IMU accelerations and orientations')
        b = 0
        out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []
        for i, l in tqdm(list(enumerate(length))):
            if l <= 12: b += l; print('\tdiscard one sequence with length', l); continue
            p = art.math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
            grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
            out_pose.append(pose[b:b + l].clone())  # N, 24, 3
            out_tran.append(tran[b:b + l].clone())  # N, 3
            out_shape.append(shape[i].clone())  # 10
            out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
            out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3
            out_vrot.append(grot[:, ji_mask])  # N, 6, 3, 3
            b += l

        print('Saving')
        amass_dir = config.processed_amass_path
        amass_dir.mkdir(exist_ok=True, parents=True)
        ds_dir = amass_dir / ds_name
        ds_dir.mkdir(exist_ok=True)

        torch.save(out_pose, ds_dir / 'pose.pt')
        torch.save(out_shape, ds_dir / 'shape.pt')
        torch.save(out_tran, ds_dir / 'tran.pt')
        torch.save(out_joint, ds_dir / 'joint.pt')
        torch.save(out_vrot, ds_dir / 'vrot.pt')
        torch.save(out_vacc, ds_dir / 'vacc.pt')
        print('Synthetic AMASS dataset is saved at', str(ds_dir))



def process_wheelposer(smooth_n = 4, split = 'fullset_am'):

    def _syn_acc(v):
        """
        Synthesize accelerations from vertex positions
        """
        mid = smooth_n // 2
        acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
        acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
        if mid != 0:
            acc[smooth_n:-smooth_n] = torch.stack(
                [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
                 for i in range(0, v.shape[0] - smooth_n * 2)])
        return acc


    vi_mask = torch.tensor([1961, 5424, 384, 3021])
    ji_mask = torch.tensor([18, 19, 15, 0])

    if split == 'am_test':
        test_split = ['S11', 'S8']
    elif split == 'am_train':
        test_split = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S12', 'S9', 'S10']
    elif split == 'am_fullset':
        test_split = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12']
    elif split == 'wu_13':
        test_split = ['S13']
    elif split == 'wu_14':
        test_split = ['S14']
    elif split == 'wu_fullset':
        test_split = ['S13', 'S14']
    else:
        return

    accs, oris, poses, trans, shapes, joints, vrots, vaccs = [], [], [], [], [], [], [], []
    body_model = art.ParametricModel(paths.smpl_file)

    if 'am' in split:
        group = 'AM'
    elif 'wu' in split:
        group = 'WU'
    else:
        return

    for subject_name in test_split:
        for motion_name in os.listdir(os.path.join(paths.raw_wheelposer_dir, group, subject_name)):
            print(subject_name, motion_name)
            path = os.path.join(paths.raw_wheelposer_dir, group, subject_name, motion_name)
            data = pickle.load(open(path, 'rb'), encoding='latin1')
            acc = torch.from_numpy(data['imu_acc']).float()
            ori = torch.from_numpy(data['imu_ori']).float()
            pose = torch.from_numpy(data['poses']).float()
            shape = torch.ones((10))
            tran = torch.zeros(pose.shape[0], 3) #discard translation

            if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                accs.append(acc.clone())
                oris.append(ori.clone())
                poses.append(pose.clone())
                trans.append(tran.clone())
                shapes.append(shape.clone()) # default shape

                p = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
                grot, joint, vert = body_model.forward_kinematics(p, shape, tran, calc_mesh=True)
                vacc = _syn_acc(vert[:, vi_mask])
                vrot = grot[:, ji_mask]
                
                joints.append(joint)
                vaccs.append(vacc)
                vrots.append(vrot)
            else:
                print('WheelPoser-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))



    # path_to_save = config.processed_wheelposer_4 / f"WheelPoser/{split}"
    path_to_save = config.processed_wheelposer_path / f"{split}"

    path_to_save.mkdir(exist_ok=True, parents=True)
    
    torch.save(poses, path_to_save / 'pose.pt')
    torch.save(shapes, path_to_save / 'shape.pt')
    torch.save(trans, path_to_save / 'tran.pt')
    torch.save(joints, path_to_save / 'joint.pt')
    torch.save(vrots, path_to_save / 'vrot.pt')
    torch.save(vaccs, path_to_save / 'vacc.pt')
    torch.save(oris, path_to_save / 'oris.pt')
    torch.save(accs, path_to_save / 'accs.pt')
    
    print('Preprocessed WheelPoser dataset is saved at', path_to_save)



if __name__ == '__main__':
    process_amass()
    # process_dipimu()
    process_wheelposer(split='am_fullset')
    process_wheelposer(split='wu_13')
    process_wheelposer(split='wu_14')
    process_wheelposer(split='wu_fullset')