import lightning as pl
import torch
import torchvision
from loguru import logger
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from .config import TrainingConfig


class CIFAR10DataModule(pl.LightningDataModule):
    """
    Handler for datasets
    """

    cifar10_normalization = torchvision.transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self._config = config

        self.train_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2
                ),
                torchvision.transforms.ToTensor(),
                self.cifar10_normalization,
            ]
        )
        self.test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                self.cifar10_normalization,
            ]
        )

    def setup(self, stage: str):
        if stage == "fit":
            logger.info("Prepare datasets")
            self._train_dataset = CIFAR10(
                "data", train=True, download=True, transform=self.train_transforms
            )
            self._validation_dataset = CIFAR10(
                "data", train=False, download=True, transform=self.test_transforms
            )
        else:
            raise NotImplementedError("Test phase data model is not implemented yet.")

    def train_dataloader(self) -> DataLoader[CIFAR10]:
        return DataLoader(
            self._train_dataset,
            batch_size=self._config.batch_size,
            num_workers=self._config.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[CIFAR10]:
        return DataLoader(
            self._validation_dataset,
            batch_size=self._config.batch_size,
            num_workers=self._config.num_workers,
            shuffle=False,
        )
