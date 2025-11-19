import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from modelgenerator.structure_tokenizer.configs.data_configs import (
    StructTokensDatasetConfig,
)
from modelgenerator.structure_tokenizer.utils.constants.residue_constants import (
    unknown_restype_idx,
)
from modelgenerator.structure_tokenizer.utils.shape_utils import (
    stack_variable_length_tensors,
)
from modelgenerator.structure_tokenizer.utils.types import PathLike

logger = logging.getLogger(__name__)


class StructTokensDataset(Dataset):
    def __init__(
        self,
        name: str,  # name of the dataset for the logs etc.
        struct_tokens_path: PathLike,
        codebook_path: PathLike,
    ) -> None:
        super().__init__()
        self.name = name
        struct_tokens_path = Path(struct_tokens_path)
        codebook_path = Path(codebook_path)
        struct_tokens = torch.load(struct_tokens_path, map_location=torch.device("cpu"))
        self.struct_tokens = [
            {
                "name": k,
                "struct_tokens": v["struct_tokens"],
                "aatype": v["aatype"],
                "residue_index": v["residue_index"],
            }
            for k, v in struct_tokens.items()
        ]
        self.codebook = torch.load(codebook_path, map_location=torch.device("cpu"))
        self._length = len(struct_tokens)

    def __getitem__(self, item) -> dict[str, str | torch.Tensor]:
        res = self.struct_tokens[item]
        embeddings = F.embedding(res["struct_tokens"], self.codebook)
        res.update({"embeddings": embeddings})
        return res

    def __len__(self) -> int:
        return self._length

    @staticmethod
    def collate_fn(
        batch: list[dict[str, torch.Tensor | str]],
    ) -> dict[str, torch.Tensor | list[str]]:
        batch_names = [b["name"] for b in batch]
        batch_struct_tokens, attention_mask = stack_variable_length_tensors(
            sequences=[b["struct_tokens"] for b in batch],
            constant_value=0,
            return_mask=True,
        )
        batch_residue_index = stack_variable_length_tensors(
            sequences=[b["residue_index"] for b in batch],
            constant_value=0,
            return_mask=False,
        )
        batch_aatype = stack_variable_length_tensors(
            sequences=[b["aatype"] for b in batch],
            constant_value=unknown_restype_idx,
            return_mask=False,
        )
        batch_embeddings = stack_variable_length_tensors(
            sequences=[b["embeddings"] for b in batch],
            constant_value=0,
            return_mask=False,
        )
        return {
            "name": batch_names,
            "embeddings": batch_embeddings,
            "aatype": batch_aatype,
            "residue_index": batch_residue_index,
            "attention_mask": attention_mask,
        }

    @classmethod
    def from_config(cls, config: StructTokensDatasetConfig) -> "StructTokensDataset":
        return cls(
            name=config.name,
            struct_tokens_path=config.struct_tokens_path,
            codebook_path=config.codebook_path,
        )
