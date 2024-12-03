import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy

from .config import TrainingConfig
from .convnext_v2 import ConvNeXt


class CifarModel(pl.LightningModule):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.model = ConvNeXt(num_blocks_list=config.model_blocks)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)
        self.log("train/loss", loss)
        self.log(f"train/accuracy", acc)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)

        if stage:
            self.log(f"{stage}/loss", loss, prog_bar=True)
            self.log(f"{stage}/accuracy", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "valid")

    def configure_optimizers(self):
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        parameters_decay, parameters_no_decay = self.model.separate_parameters()

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in parameters_decay],
                "weight_decay": 0.01,
            },
            {
                "params": [param_dict[pn] for pn in parameters_no_decay],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.config.optimisation_config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.01,
            amsgrad=False,
        )

        steps_per_epoch = 50200 // self.config.batch_size
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=self.config.optimisation_config.learning_rate,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }

        self.first_cnn = self.model[0][0]
        self.model = torch.compile(self.model)
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    def on_after_backward(self):
        global_step = self.global_step
        for idx, weight in enumerate(self.first_cnn.parameters()):
            self.logger.experiment.add_histogram(f"cnn_{idx}", weight.grad, global_step)
            self.log(f"gradient/cnn_{idx}", weight.grad.abs().mean())
