import os
from typing import Optional

import lightning as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.loggers.logger import Logger
from loguru import logger

from .baseline import CifarModel
from .config import TrainingConfig
from .dataset import CIFAR10DataModule


class CIFAR10Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.trainer: Optional[pl.Trainer] = None

    def train(self):
        lr_monitor = LearningRateMonitor(logging_interval="step")
        early_stop_callback = EarlyStopping(
            monitor=self.config.early_stop_metric,
            min_delta=self.config.early_stop_metric_min_delta,
            patience=self.config.optimisation_config.early_stop_patience,
            verbose=True,
            mode=self.config.early_stop_metric_mode,
        )
        weight_dir = os.path.join(
            self.config.logging_dir, "model", self.config.experiment_name, "weights"
        )
        model_checkpoints = ModelCheckpoint(
            dirpath=weight_dir,
            monitor=self.config.model_checkpoint_metric,
            save_top_k=self.config.model_checkpoint_metric_save_top,
            mode=self.config.model_checkpoint_metric_mode,
            filename="epoch={epoch:02d}-={"
            + f"{self.config.model_checkpoint_metric}"
            + ":.4f}",
            save_weights_only=False,
            auto_insert_metric_name=False,  # auto-inserting cause creating a folder when metric has slash
            save_last=True,
        )

        tensorboard_logger = TensorBoardLogger(
            os.path.join(self.config.logging_dir, "tensorboard"),
            self.config.experiment_name,
            version="tb",
        )
        loggers: list[Logger] = [tensorboard_logger]
        if self.config.use_wandb_logger:
            wandb_logger = WandbLogger(
                project=self.config.wandb_name,
                name=self.config.experiment_name,
                save_dir=os.path.join(self.config.logging_dir, "wandb"),
            )
            loggers.append(wandb_logger)
            wandb_logger.experiment.log_code(root=".")

        self.trainer = pl.Trainer(
            max_epochs=self.config.num_epochs + 1,
            callbacks=[lr_monitor, early_stop_callback, model_checkpoints],
            logger=loggers,
            precision=self.config.precision,  # type:ignore
            benchmark=True,
            deterministic=False,
            default_root_dir=self.config.logging_dir,
            log_every_n_steps=self.config.log_every_n_steps,
            accumulate_grad_batches=self.config.optimisation_config.grad_accumulation,
            gradient_clip_val=self.config.gradient_clip,
            devices=self.config.num_devices,
            val_check_interval=self.config.validation_check_interval,
        )

        data_model = CIFAR10DataModule(config=self.config)
        model = CifarModel(self.config)

        logger.info("Start training")
        self.trainer.fit(model, data_model)
        logger.info("Training finalized")
