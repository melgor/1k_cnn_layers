import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from loguru import logger
from omegaconf import OmegaConf


@dataclass
class OptimisationConfig:
    early_stop_patience: int = 5
    learning_rate: float = 0.001
    scheduler_interval: str = "epoch"
    grad_accumulation: int = 1


@dataclass
class TrainingConfig:
    experiment_name: Optional[str] = None
    logging_dir: Optional[str] = None
    model_blocks: Optional[list[int]] = None
    wandb_name: Optional[str] = "1k_layer"
    batch_size: int = 512
    num_epochs: int = 100
    precision: str = "bf16-mixed"
    optimisation_config: OptimisationConfig = field(
        default_factory=lambda: OptimisationConfig()
    )
    log_every_n_steps: int = 10
    gradient_clip: float = 5.0
    num_devices: int = 1
    seed: int = 42
    pin_memory: bool = True
    num_workers: int = 8
    validation_check_interval: float = 1.0
    early_stop_metric: str = "valid/accuracy"
    early_stop_metric_min_delta: float = 0.01
    early_stop_metric_mode: str = "max"
    model_checkpoint_metric: str = "valid/accuracy"
    model_checkpoint_metric_mode: str = "max"
    model_checkpoint_metric_save_top: int = 2
    use_wandb_logger: bool = False


def validate_config_and_init_paths(config: TrainingConfig):
    today = datetime.now()
    config.experiment_name = (
        f"{config.experiment_name}_{today.strftime('%Y_%m_%d_%h_%H_%M_%s')}"
    )
    assert config.experiment_name is not None, "Set experiment_name"
    assert config.logging_dir is not None, "Set logging_dir"
    assert config.model_blocks is not None, "Set model_blocks"
    assert (
        len(config.model_blocks) == 4
    ), f"Should be 4 numbers in blocks, now it is {len(config.model_blocks)}"


def load_config(config_path: Optional[str], load_args: bool = False) -> TrainingConfig:
    # Load the configuration file and merge it with the default configuration
    logger.info(f"Load config: {config_path}")
    default_config = TrainingConfig()  # type: ignore
    default_config_omega: TrainingConfig = OmegaConf.structured(default_config)

    if config_path is not None:
        user_config = OmegaConf.load(config_path)
        all_configs = [default_config_omega, user_config]
        if load_args:
            # overwrite any argument by command line. To make it work, remove path to base config file from sys.argv
            sys.argv.pop(1)
            cli_config = OmegaConf.from_cli()
            all_configs.append(cli_config)

        default_config_omega = OmegaConf.merge(*all_configs)  # type: ignore
        validate_config_and_init_paths(default_config_omega)

    return default_config_omega


if __name__ == "__main__":
    config_path = None
    print(len(sys.argv))
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    config_omega = load_config(config_path)
    print(OmegaConf.to_yaml(config_omega))
