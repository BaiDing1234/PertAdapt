import logging
from typing import Any

import torch
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT

from modelgenerator.structure_tokenizer.models.structure_tokenizer import (
    StructureTokenizerModel,
)

logger = logging.getLogger(__name__)


class StructureTokenizerLightning(L.LightningModule):
    def __init__(self, pretrained_model_name_or_path: str):
        super().__init__()
        self.model = StructureTokenizerModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )

    def forward(
        self,
        atom_positions: torch.Tensor,
        aatype: torch.Tensor,
        attention_mask: torch.Tensor,
        residue_index: torch.Tensor,
        num_recycles: int | None = None,
    ) -> dict[str, torch.Tensor]:
        output = self.model.forward(
            atom_positions=atom_positions,
            aatype=aatype,
            attention_mask=attention_mask,
            residue_index=residue_index,
            num_recycles=num_recycles,
        )
        return output

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        raise NotImplementedError("StructureTokenizerLightning can not be trained.")

    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        raise NotImplementedError("StructureTokenizerLightning can not be validated.")

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        raise NotImplementedError("StructureTokenizerLightning can not be tested.")

    def predict_step(
        self, batch, batch_idx, dataloader_idx=0
    ) -> dict[str, torch.Tensor]:
        atom_positions = batch["atom_positions"]
        aatype = batch["aatype"]
        attention_mask = batch["attention_mask"]
        residue_index = batch["residue_index"]
        output = self.forward(
            atom_positions=atom_positions,
            aatype=aatype,
            attention_mask=attention_mask,
            residue_index=residue_index,
            num_recycles=None,
        )
        return output

    def freeze_unused_params(self) -> None:
        self.model.freeze_unused_params()
