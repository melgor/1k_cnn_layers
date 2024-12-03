import sys

import torch

from src.config import load_config
from src.trainer import CIFAR10Trainer

torch.backends.cuda.matmul.allow_tf32 = True

if __name__ == "__main__":
    config_path = sys.argv[1]
    config = load_config(config_path)
    trainer = CIFAR10Trainer(config)
    trainer.train()
