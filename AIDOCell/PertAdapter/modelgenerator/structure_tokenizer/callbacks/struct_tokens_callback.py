import functools
import operator
from pathlib import Path
from typing import Literal

import lightning as L
import torch

from lightning.pytorch.callbacks import BasePredictionWriter

from modelgenerator.structure_tokenizer.models import EquiformerEncoderLightning


class StructTokensCallback(BasePredictionWriter):
    def __init__(
        self,
        output_dir: Path | str,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"],
    ):
        assert write_interval == "epoch", "only write_interval='epoch' is supported"
        super().__init__(write_interval=write_interval)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predict_id_names: dict[int, str] | None = None

    def setup(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        assert isinstance(
            pl_module, EquiformerEncoderLightning
        ), "pl_module must be EquiformerEncoderLightning"
        config = trainer.datamodule.config
        self.predict_id_names = config.dataloader_idx_to_name

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        if len(self.predict_id_names) == 1:  # only one dataset
            predictions = functools.reduce(operator.or_, predictions)
            predictions = {
                name: {k: v.to("cpu") for k, v in d.items()}
                for name, d in predictions.items()
            }
            torch.save(
                predictions,
                self.output_dir / f"{self.predict_id_names[0]}_struct_tokens.pt",
            )
        else:  # several datasets
            for dataloader_idx, preds in enumerate(predictions):
                predictions = functools.reduce(operator.or_, preds)
                predictions = {
                    name: {k: v.to("cpu") for k, v in d.items()}
                    for name, d in predictions.items()
                }
                torch.save(
                    predictions,
                    self.output_dir
                    / f"{self.predict_id_names[dataloader_idx]}_struct_tokens.pt",
                )
        torch.save(trainer.model.encoder.codebook, self.output_dir / "codebook.pt")
