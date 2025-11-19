from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


@dataclass
class ValidationExperimentConfig:
    name: str  # name of the project
    wandb_project: str
    output_dir: str | Path  # runs directory
    path_ckpt: str | Path
    device: Device
    log_every_n_steps: int
    loggers: list[Any]
    devices: Any = "auto"  # list[int] | str | int
    seed: int = 0  # not necessary if no cropping


@dataclass
class EncodingConfig:
    folder_name: str  # folder name
    output_dir: str | Path  # output directory
    path_ckpt: str | Path
    device: Device
    return_predictions: bool
    devices: Any = "auto"  # list[int] | str | int
    seed: int = 0  # not necessary if no cropping


@dataclass
class DecodingConfig:
    folder_name: str  # folder name
    output_dir: str | Path  # output directory
    path_ckpt: str | Path
    device: Device
    devices: Any = "auto"  # list[int] | str | int
    seed: int = 0  # not necessary ?
