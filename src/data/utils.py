r"""
Dataset util functions
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.AMASS import AMASS
from src.data.WHEELPOSER import WHEELPOSER
from src.data.DIP import DIP


def train_val_split(dataset, train_pct):
    # get the train and val split
    total_size = len(dataset)
    train_size = int(train_pct * total_size)
    val_size = total_size - train_size
    return train_size, val_size

def get_dataset(config=None, test_only=False, fine_tune=False):
    model = config.model
    # load the dataset
    print("loading dataset for", model)

    if model == "IMUPoser_WheelPoser_AMASS":
        if not test_only:
            train_dataset = AMASS("train", config, stage="full")
        test_dataset = AMASS("test", config, stage="full")
    elif model == "IMUPoser_WheelPoser_DIP":
        if not test_only:
            train_dataset = DIP("train", config, stage="full")
        test_dataset = DIP("test", config, stage="full")
    elif model == "IMUPoser_WheelPoser_WHEELPOSER":
        if not test_only:
            train_dataset = WHEELPOSER("train", config, stage="full")
        test_dataset = WHEELPOSER("test", config, stage="full")

    elif model == "TIP_WheelPoser_AMASS":
        if not test_only:
            train_dataset = AMASS("train", config, stage="full")
        test_dataset = AMASS("test", config, stage="full")
    elif model == "TIP_WheelPoser_WHEELPOSER":
        if not test_only:
            train_dataset = WHEELPOSER("train", config, stage="full")
        test_dataset = WHEELPOSER("test", config, stage="full")

    #IMU2Leaf
    elif model == "IMU2Leaf_WheelPoser_AMASS":
        if not test_only:
            train_dataset = AMASS("train", config, stage="imu2leaf")
        test_dataset = AMASS("test", config, stage="imu2leaf")
    elif model == "IMU2Leaf_WheelPoser_DIP":
        if not test_only:
            train_dataset = DIP("train", config, stage="imu2leaf")
        test_dataset = DIP("test", config, stage="imu2leaf")
    elif model == "IMU2Leaf_WheelPoser_WHEELPOSER":
        if not test_only:
            train_dataset = WHEELPOSER("train", config, stage="imu2leaf")
        test_dataset = WHEELPOSER("test", config, stage="imu2leaf")

    #Leaf2Full
    elif model == "Leaf2Full_WheelPoser_AMASS":
        if not test_only:
            train_dataset = AMASS("train", config, stage="leaf2full")
        test_dataset = AMASS("test", config, stage="leaf2full")
    elif model == "Leaf2Full_WheelPoser_DIP":
        if not test_only:
            train_dataset = DIP("train", config, stage="leaf2full")
        test_dataset = DIP("test", config, stage="leaf2full")
    elif model == "Leaf2Full_WheelPoser_WHEELPOSER":
        if not test_only:
            train_dataset = WHEELPOSER("train", config, stage="leaf2full")
        test_dataset = WHEELPOSER("test", config, stage="leaf2full")
    
    #Full2Pose
    elif model == "Full2Pose_WheelPoser_AMASS":
        if not test_only:
            train_dataset = AMASS("train", config, stage="full2pose")
        test_dataset = AMASS("test", config, stage="full2pose")
    elif model == "Full2Pose_WheelPoser_DIP":
        if not test_only:
            train_dataset = DIP("train", config, stage="full2pose")
        test_dataset = DIP("test", config, stage="full2pose")
    elif model == "Full2Pose_WheelPoser_WHEELPOSER":
        if not test_only:
            train_dataset = WHEELPOSER("train", config, stage="full2pose")
        test_dataset = WHEELPOSER("test", config, stage="full2pose")

    else:
        print("Enter a valid model")
        return

    if not test_only:
        # get the train and val split
        train_size, val_size = train_val_split(train_dataset, train_pct=config.train_pct)

        # split the dataset
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    if not test_only:
        return train_dataset, test_dataset, val_dataset
    else:
        return test_dataset

def get_datamodule(config):
    model = config.model
    # load the dataset
    # if model in ["IMUPoser_WheelPoser_AMASS", "IMUPoser_WheelPoser_DIP", "IMUPoser_WheelPoser_WHEELPOSER"]:
    #     return WheelPoserDataModule(config)
    # else:
    #     print("Enter a valid model")
    return WheelPoserDataModule(config)

def pad_seq(batch):
    inputs = [item[0] for item in batch]
    outputs = [item[1] for item in batch]
    
    input_lens = [item.shape[0] for item in inputs]
    output_lens = [item.shape[0] for item in outputs]
    
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    outputs = nn.utils.rnn.pad_sequence(outputs, batch_first=True)
    return inputs, outputs, input_lens, output_lens

class WheelPoserDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        self.train_dataset, self.test_dataset, self.val_dataset = get_dataset(self.config)
        print("Done with setup")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, collate_fn=pad_seq, num_workers=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.batch_size, collate_fn=pad_seq, num_workers=8, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.batch_size, collate_fn=pad_seq, num_workers=8, shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.batch_size, collate_fn=pad_seq, num_workers=8, shuffle=False)
