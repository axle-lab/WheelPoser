import torch
from tqdm import tqdm

from src.config import Config
from src.articulate import ParametricModel
from src.articulate import math

config = Config(project_root_dir="./")

target_fps = 60

def smooth_avg(acc=None, s=3):
    nan_tensor = (torch.zeros((s // 2, acc.shape[1], acc.shape[2])) * torch.nan)
    acc = torch.cat((nan_tensor, acc, nan_tensor))
    tensors = []
    for i in range(s):
        L = acc.shape[0]
        tensors.append(acc[i:L-(s-i-1)])

    smoothed = torch.stack(tensors).nanmean(dim=0)
    return smoothed

def _resample(tensor, target_fps):
    r"""
        Resample to the target fps, assumes 60fps input
    """
    indices = torch.arange(0, tensor.shape[0], 60/target_fps)

    start_indices = torch.floor(indices).long()
    end_indices = torch.ceil(indices).long()
    end_indices[end_indices >= tensor.shape[0]] = tensor.shape[0] - 1 # handling edge cases

    start = tensor[start_indices]
    end = tensor[end_indices]
    
    floats = indices - start_indices
    for shape_index in range(len(tensor.shape) - 1):
        floats = floats.unsqueeze(1)
    weights = torch.ones_like(start) * floats
    torch_lerped = torch.lerp(start, end, weights)
    return torch_lerped

path_to_save = config.combined_amass_path
path_to_save.mkdir(exist_ok=True, parents=True)

# process AMASS first
for fpath in (config.processed_amass_path).iterdir():
    joint = torch.load(fpath / "joint.pt")
    shape = torch.load(fpath / "shape.pt")
    tran = torch.load(fpath / "tran.pt")
    vacc = torch.load(fpath / "vacc.pt")
    vrot = torch.load(fpath / "vrot.pt")
    pose = [math.axis_angle_to_rotation_matrix(x).view(-1,24,3,3) for x in torch.load(fpath / "pose.pt")]
    
    # save the data
    fdata = {
        "joint": joint,
        "pose": pose,
        "shape": shape,
        "tran": tran,
        "acc": vacc,
        "ori": vrot
    }
    
    torch.save(fdata, path_to_save / f"{fpath.name}.pt")

#Update the path to save
path_to_save = config.combined_wheelposer_path
path_to_save.mkdir(exist_ok=True, parents=True)

# process WheelPoser
for fpath in (config.processed_wheelposer_path).iterdir():
    joint = torch.load(fpath / "joint.pt")
    shape = torch.load(fpath / "shape.pt")
    tran = torch.load(fpath / "tran.pt")
    acc = torch.load(fpath / "accs.pt")
    rot = torch.load(fpath / "oris.pt")
    pose = [math.axis_angle_to_rotation_matrix(x).view(-1,24,3,3)  for x in torch.load(fpath / "pose.pt")]
    # save the data
    fdata = {
        "joint": joint,
        "pose": pose,
        "shape": shape,
        "tran": tran,
        "acc": acc,
        "ori": rot
    }
    
    torch.save(fdata, path_to_save / f"{fpath.name}.pt")
