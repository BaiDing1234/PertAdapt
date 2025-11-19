import logging

import lightning as L
import torch
from lightning.pytorch.utilities import CombinedLoader
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader

from modelgenerator.structure_tokenizer.configs.data_configs import (
    StructTokensDataConfig,
)
from modelgenerator.structure_tokenizer.datasets.struct_tokens_dataset import (
    StructTokensDataset,
)

logger = logging.getLogger(__name__)


class EmptyDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError("Empty dataset cannot be indexed")


class StructTokensLightningDataModule(L.LightningDataModule):
    def __init__(
        self,
        config: StructTokensDataConfig,
    ):
        super().__init__()
        self.config = config
        self.struct_tokens_datasets: dict[str, StructTokensDataset] | None = None
        self.num_workers = config.num_workers

    def setup(self, stage: str) -> None:
        """
        stage: fit/validate/test/predict stage
        """
        match stage:
            case "fit":
                raise NotImplementedError("fit stage is not implemented for decoder.")
            case "validate":
                raise NotImplementedError(
                    "validate stage is not implemented for decoder."
                )
            case "test":
                raise NotImplementedError("test stage is not implemented for decoder.")
            case "predict":
                self.struct_tokens_datasets = {
                    name: StructTokensDataset.from_config(config=config)
                    for name, config in self.config.configs.items()
                }
            case _:
                raise NotImplementedError

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        if self.struct_tokens_datasets:
            dataloaders = {
                name: DataLoader(
                    dataset=dataset,
                    batch_size=self.config.configs[name].batch_size,
                    collate_fn=StructTokensDataset.collate_fn,
                    num_workers=self.num_workers,
                    shuffle=False,
                    sampler=None,
                    drop_last=False,
                    persistent_workers=True if self.num_workers > 0 else False,
                )
                for name, dataset in self.struct_tokens_datasets.items()
            }
            combined_loader = CombinedLoader(dataloaders, mode="sequential")
            return combined_loader
        else:
            return DataLoader(EmptyDataset())

    def teardown(self, stage: str) -> None:
        """
        stage: fit/val/test/predict stage
        """
        self.struct_tokens_datasets = None
