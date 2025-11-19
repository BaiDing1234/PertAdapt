import logging

import lightning as L
import torch
from lightning.pytorch.overrides.distributed import UnrepeatedDistributedSampler
from lightning.pytorch.utilities import CombinedLoader
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader

from modelgenerator.structure_tokenizer.configs.data_configs import ProteinDataConfig
from modelgenerator.structure_tokenizer.datasets.protein_dataset import (
    ProteinCSVParquetDataset,
    ProteinDataset,
)

logger = logging.getLogger(__name__)


class EmptyDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError("Empty dataset cannot be indexed")


class ProteinLightningDataModule(L.LightningDataModule):
    def __init__(
        self,
        config: ProteinDataConfig,
    ):
        super().__init__()
        self.config = config
        self.proteins_datasets: dict[str, ProteinDataset] | None = None
        self.num_workers = config.num_workers

    def setup(self, stage: str) -> None:
        """
        stage: fit/validate/test/predict stage
        """
        match stage:
            case "fit":
                raise NotImplementedError(
                    "fit stage is not implemented for structure_tokenizer."
                )
            case "validate":
                raise NotImplementedError(
                    "validate stage is not implemented for structure_tokenizer."
                )
            case "test":
                raise NotImplementedError(
                    "test stage is not implemented for structure_tokenizer."
                )
            case "predict":
                self.proteins_datasets = {
                    name: ProteinCSVParquetDataset.from_config(
                        protein_dataset_config=config
                    )
                    for name, config in self.config.configs.items()
                }
            case _:
                raise NotImplementedError

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        if self.proteins_datasets:
            dataloaders = {
                name: DataLoader(
                    dataset=dataset,
                    batch_size=self.config.configs[name].batch_size,
                    collate_fn=ProteinDataset.collate_fn,
                    num_workers=self.num_workers,
                    shuffle=False,
                    sampler=None,
                    drop_last=False,
                    persistent_workers=True if self.num_workers > 0 else False,
                )
                for name, dataset in self.proteins_datasets.items()
            }
            combined_loader = CombinedLoader(dataloaders, mode="sequential")
            return combined_loader
        else:
            return DataLoader(EmptyDataset())

    def teardown(self, stage: str) -> None:
        """
        stage: fit/val/test/predict stage
        """
        self.proteins_datasets = None
