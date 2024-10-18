import argparse
import torch
import pickle as pkl
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from src.config import Config, joint_set
from src.evaluator import ReducedPoseEvaluator
from src.models.utils import get_model
from src.data.utils import get_datamodule
from src.articulate.math import *
from src.utils import get_checkpoints



# set the random seed
seed_everything(42, workers = True)


model_names = ["Leaf2Full_WheelPoser_AMASS"]
experiment_names = "3_Stage_500"
max_epochs = 500

leave_one_out_exp = True

# %%
best_ckpts = get_checkpoints(model_names, experiment_names)

# %%
best_ckpts

# %%
# load model
pretrained_config = Config(experiment=experiment_names, model="Leaf2Full_WheelPoser_AMASS", project_root_dir=".", joints_set=joint_set.WheelPoser, pred_joints_set=joint_set.upper_body,
                normalize=True, r6d=True, loss_type="mse", use_joint_loss=False, mkdir=False, upper_body_only=True)
pretrained_model = get_model(pretrained_config).load_from_checkpoint(best_ckpts[model_names[0]], config=pretrained_config)

# %%
# setup Config
config = Config(experiment=experiment_names, model="Leaf2Full_WheelPoser_WHEELPOSER", project_root_dir=".", joints_set=joint_set.WheelPoser, pred_joints_set=joint_set.upper_body,
                normalize=True, r6d=True, loss_type="mse", use_joint_loss=False, upper_body_only=False, exp_setup='leave_13_out', upsample_copies=7)

# %%
# instantiate model and datamodule
model = get_model(config, pretrained_model)
datamodule = get_datamodule(config)
checkpoint_path = config.checkpoint_path


# %%
wandb_logger = WandbLogger(project=config.experiment, save_dir=checkpoint_path, name=config.model)

early_stopping_callback = EarlyStopping(monitor="validation_step_loss", mode="min", verbose=False,
                                        min_delta=0.00001, patience=50)
checkpoint_callback = ModelCheckpoint(monitor="validation_step_loss", mode="min", verbose=False, 
                                      save_top_k=5, dirpath=checkpoint_path, save_weights_only=True, 
                                      filename='epoch={epoch}-val_loss={validation_step_loss:.5f}')

trainer = pl.Trainer(fast_dev_run=False, logger=wandb_logger, max_epochs=max_epochs, accelerator="gpu", devices=[0],
                     callbacks=[early_stopping_callback, checkpoint_callback], deterministic=False)

# %%
trainer.fit(model, datamodule=datamodule)

# %%
# log this
# checkpoint_callback.best_k_models, checkpoint_callback.best_model_path

with open(checkpoint_path / "best_model.txt", "w") as f:
    f.write(f"{checkpoint_callback.best_model_path}\n\n{checkpoint_callback.best_k_models}")
