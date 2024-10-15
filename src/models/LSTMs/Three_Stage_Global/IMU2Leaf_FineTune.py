import torch.nn as nn
import torch
import pytorch_lightning as pl
from src.models.LSTMs.RNN import RNN
from src.models.loss_functions import *

class IMU2Leaf_WheelPoser_FineTune(pl.LightningModule):
    r"""
    3 Stage Pose S1
    Inputs - N IMUs, Outputs - SMPL Pose params (in Rot Matrix)
    """
    def __init__(self, config, pretrained_model):
        super().__init__()
        n_input = 12 * len(config.joints_set)
        self.batch_size = config.batch_size
        # self.dip_model = RNN(n_input=n_input, n_output=3 * 3, n_hidden=256, bidirectional=True)
        self.pretrained_model = pretrained_model

        if config.loss_type == "l1":
            self.loss = nn.L1Loss()
        elif config.loss_type == "weighted_mse":
            weights = torch.ones(216).reshape((24, 9)) * 2
            amass_joints = torch.Tensor([18, 19, 1, 2, 15, 0])
            weighted_joints = amass_joints[config.joints_set].long()
            weights[weighted_joints] = 1 # give a weight of 1 to the joints that have imus
            weights = weights.reshape((216))

            self.loss = weighted_mse(weights=weights)
        else:
            self.loss = nn.MSELoss()

        self.lr = 0.001
        self.save_hyperparameters(ignore=['pretrained_model'])

    def forward(self, imu_inputs, imu_lens):
        pred_pose = self.pretrained_model(imu_inputs, imu_lens)
        return pred_pose

    def training_step(self, batch, batch_idx):
        imu_inputs, target_pose, input_lengths, _ = batch

        pred_pose = self(imu_inputs, input_lengths)
        loss = self.loss(pred_pose, target_pose)
        self.log(f"training_step_loss", loss.item(), batch_size=self.batch_size)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        imu_inputs, target_pose, input_lengths, _ = batch

        pred_pose = self(imu_inputs, input_lengths)
        loss = self.loss(pred_pose, target_pose)
        self.log(f"validation_step_loss", loss.item(), batch_size=self.batch_size)

        return {"loss": loss}

    def predict_step(self, batch, batch_idx):
        imu_inputs, target_pose, input_lengths, _ = batch

        pred_pose = self(imu_inputs, input_lengths)
        loss = self.loss(pred_pose, target_pose)

        return {"loss": loss.item(), "pred": pred_pose, "true": target_pose}

    def training_epoch_end(self, outputs):
        self.epoch_end_callback(outputs, loop_type="train")

    def validation_epoch_end(self, outputs):
        self.epoch_end_callback(outputs, loop_type="val")

    def test_epoch_end(self, outputs):
        self.epoch_end_callback(outputs, loop_type="test")

    def epoch_end_callback(self, outputs, loop_type="train"):
        loss = []
        for output in outputs:
            # stuff to do here:
            # pass the true and predicted params through the SMPL Torch model and get the vertices
            # compute the mesh error from there

            loss.append(output["loss"])

        # agg the losses
        avg_loss = torch.mean(torch.Tensor(loss))
        self.log(f"{loop_type}_loss", avg_loss, prog_bar=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
