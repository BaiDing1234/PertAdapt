# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# copied and modified from esmfold's codebase
# https://github.com/facebookresearch/esm/blob/main/esm/esmfold/v1/esmfold.py
from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from lightning.pytorch.utilities.types import STEP_OUTPUT
from openfold.data.data_transforms import make_atom14_masks
from openfold.utils.feats import atom14_to_atom37
from openfold.utils.loss import compute_tm, compute_predicted_aligned_error
import lightning as L

from modelgenerator.structure_tokenizer.configs.models_configs import (
    ESMFoldDecoderConfig,
)
from modelgenerator.structure_tokenizer.layers.esmfold.categorical_mixture import (
    categorical_lddt,
)
from modelgenerator.structure_tokenizer.layers.esmfold.trunk import FoldingTrunk
from modelgenerator.structure_tokenizer.utils.constants.structure_tokenizer import (
    N_TOKENS,
    LDDT_BINS,
    DISTOGRAM_BINS,
)
from modelgenerator.structure_tokenizer.utils.init_params import (
    init_params_recursively_,
)
from modelgenerator.structure_tokenizer.utils.misc import get_config_from_dict


class ESMFoldDecoder(
    nn.Module,
    PyTorchModelHubMixin,
    # library_name="",
    # repo_url="",
    # docs_url="",
    # license="",
    tags=["biology", "genbio", "protein"],
):
    def __init__(self, config: ESMFoldDecoderConfig):
        super().__init__()

        self.config = get_config_from_dict(
            config_dict=asdict(config), config=ESMFoldDecoderConfig
        )

        c_s = self.config.folding_trunk.sequence_state_dim
        c_z = self.config.folding_trunk.pairwise_state_dim

        self.mlp_quantize_to_dec = nn.Sequential(
            nn.Linear(self.config.quantize_dim, c_s),
            nn.SiLU(),
            nn.Linear(c_s, c_s),
        )

        self.trunk: FoldingTrunk = FoldingTrunk(config=self.config.folding_trunk)

        self.distogram_head = nn.Linear(c_z, DISTOGRAM_BINS)
        self.ptm_head = nn.Linear(c_z, DISTOGRAM_BINS)
        self.lm_head = nn.Linear(c_s, N_TOKENS)
        self.lddt_head = nn.Sequential(
            nn.LayerNorm(self.config.folding_trunk.structure_module.c_s),
            nn.Linear(
                self.config.folding_trunk.structure_module.c_s,
                self.config.lddt_head_hidden_dim,
            ),
            nn.Linear(
                self.config.lddt_head_hidden_dim, self.config.lddt_head_hidden_dim
            ),
            nn.Linear(self.config.lddt_head_hidden_dim, 37 * LDDT_BINS),
        )

    def reset_parameters(self) -> None:
        init_params_recursively_(self)

    def forward(
        self,
        s_s_0: torch.Tensor,  # [B, L, Cs]
        aatype: torch.Tensor,  # [B, L]
        attention_mask: torch.Tensor | None = None,  # [B, L]
        residue_index: torch.Tensor | None = None,  # [B, L]
        num_recycles: int | None = None,  # int
    ):
        """Runs a forward pass given input tokens.
        Use `model.infer` to run inference from a sequence.

        Args:
            s_s_0 (torch.Tensor): embeddings from the encoder.
            aatype (torch.Tensor): Tensor containing indices corresponding to amino acids. Indices match
                openfold.np.residue_constants.restype_order_with_x.
            attention_mask (torch.Tensor): Binary tensor with 1 meaning position is unmasked and 0 meaning position is masked.
            residue_index (torch.Tensor): Residue indices of amino acids. Will assume contiguous if not provided.
            num_recycles (int): How many recycle iterations to perform. If None, defaults to training max
                recycles, which is 3.

        Returns (batch size 2, length 500)
            {'frames': torch.Size([8, 2, 500, 7]),
             'sidechain_frames': torch.Size([8, 2, 500, 8, 4, 4]),
             'unnormalized_angles': torch.Size([8, 2, 500, 7, 2]),
             'angles': torch.Size([8, 2, 500, 7, 2]),
             'positions': torch.Size([8, 2, 500, 14, 3]),
             'states': torch.Size([8, 2, 500, 384]),
             's_s': torch.Size([2, 500, 768]),
             's_z': torch.Size([2, 500, 500, 128]),
             'distogram_logits': torch.Size([2, 500, 500, 64]),
             'lm_logits': torch.Size([2, 500, 24]),
             'aatype': torch.Size([2, 500]),
             'atom14_atom_exists': torch.Size([2, 500, 14]),
             'residx_atom14_to_atom37': torch.Size([2, 500, 14]),
             'residx_atom37_to_atom14': torch.Size([2, 500, 37]),
             'atom37_atom_exists': torch.Size([2, 500, 37]),
             'residue_index': torch.Size([2, 500]),
             'lddt_head': torch.Size([8, 2, 500, 37, 50]),
             'plddt': torch.Size([2, 500, 37]),
             'ptm_logits': torch.Size([2, 500, 500, 64]),
             'ptm': torch.Size([2]),
             'aligned_confidence_probs': torch.Size([2, 500, 500, 64]),
             'predicted_aligned_error': torch.Size([2, 500, 500]),
             'max_predicted_aligned_error': torch.Size([]),
             'coords': torch.Size([2, 500, 37, 3]),
             "qx": qx,
             }
        """

        B, L = s_s_0.shape[:2]

        # Project to the trunk dimension
        s_s_0 = self.mlp_quantize_to_dec(s_s_0)
        qx = s_s_0.clone()  # to keep track

        # Convert s_s_0 to the precision used by the trunk and
        # the structure module. These tensors may be a lower precision if, for example,
        # we're running the language model in fp16 precision.
        s_s_0 = s_s_0.to(self.trunk.trunk2sm_s.weight.dtype)
        s_z_0 = s_s_0.new_zeros(
            B, L, L, self.config.folding_trunk.pairwise_state_dim
        )  # [B, L, L, Cz=128]

        structure: dict = self.trunk(
            seq_feats=s_s_0,
            pair_feats=s_z_0,
            true_aa=aatype,
            residx=residue_index,
            mask=attention_mask,
            no_recycles=num_recycles,
        )
        # Documenting what we expect:
        structure = {
            k: v
            for k, v in structure.items()
            if k
            in [
                "s_z",  # [B, L, L, Cz=128]
                "s_s",  # [B, L, Cs=1024]
                "frames",  # [nSmLayers=8, B, L, 7]
                "sidechain_frames",  # [nSmLayers=8, B, L, 8, 4, 4]
                "unnormalized_angles",  # [nSmLayers=8, B, 7, 2]
                "angles",  # [nSmLayers=8, B, 7, 2]
                "positions",  # [nSmLayers=8, B, L, 14, 3]
                "states",  # [nSmLayers=8, B, L, dSM=384]
            ]
        }

        disto_logits = self.distogram_head(structure["s_z"])
        disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2
        structure["distogram_logits"] = disto_logits

        lm_logits = self.lm_head(structure["s_s"])  # [B, L, nTokens=23]
        structure["lm_logits"] = lm_logits

        structure["aatype"] = aatype
        make_atom14_masks(structure)

        for k in [
            "atom14_atom_exists",
            "atom37_atom_exists",
        ]:
            structure[k] *= attention_mask.unsqueeze(-1)
        structure["residue_index"] = residue_index

        lddt_head = self.lddt_head(structure["states"]).reshape(
            structure["states"].shape[0], B, L, -1, LDDT_BINS
        )
        structure["lddt_head"] = lddt_head
        plddt = categorical_lddt(lddt_head[-1], bins=LDDT_BINS)
        structure["plddt"] = (
            100 * plddt
        )  # we predict plDDT between 0 and 1, scale to be between 0 and 100.

        ptm_logits = self.ptm_head(structure["s_z"])

        seqlen = attention_mask.type(torch.int64).sum(1)
        structure["ptm_logits"] = ptm_logits
        structure["ptm"] = torch.stack(
            [
                compute_tm(
                    batch_ptm_logits[None, :sl, :sl],
                    max_bins=31,
                    no_bins=DISTOGRAM_BINS,
                )
                for batch_ptm_logits, sl in zip(ptm_logits, seqlen)
            ]
        )
        structure.update(
            compute_predicted_aligned_error(
                ptm_logits, max_bin=31, no_bins=DISTOGRAM_BINS
            )
        )
        structure["coords"] = atom14_to_atom37(structure["positions"][-1], structure)
        structure["qx"] = qx
        return structure

    def freeze_unused_params(self) -> list[str]:
        return self.freeze_lddt_lm_ptm_head()

    def freeze_lddt_lm_ptm_head(self) -> list[str]:
        # freeze lddt_head, lm_head, ptm_head because they are not used in computing the loss
        frozen = []
        for name, params in self.named_parameters():
            if "lddt_head" in name:
                params.requires_grad_(False)
            elif "lm_head" in name:
                params.requires_grad_(False)
            elif "ptm_head" in name:
                params.requires_grad_(False)
            else:
                continue
            frozen.append(name)
        return frozen


class ESMFoldDecoderLightning(L.LightningModule):
    def __init__(self, pretrained_model_name_or_path: str) -> None:
        super().__init__()
        self.decoder = ESMFoldDecoder.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )

    def forward(
        self,
        s_s_0: torch.Tensor,  # [B, L, Cs]
        aatype: torch.Tensor,  # [B, L]
        attention_mask: torch.Tensor | None = None,  # [B, L]
        residue_index: torch.Tensor | None = None,  # [B, L]
        num_recycles: int | None = None,  # int
    ) -> dict[str, torch.Tensor]:
        assert not self.training, "Only inference mode is supported."
        return self.decoder(
            s_s_0=s_s_0,
            aatype=aatype,
            attention_mask=attention_mask,
            residue_index=residue_index,
            num_recycles=num_recycles,
        )

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        raise NotImplementedError("ESMFoldDecoderLightning can not be trained.")

    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        raise NotImplementedError("ESMFoldDecoderLightning can not be validated.")

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        raise NotImplementedError("ESMFoldDecoderLightning can not be tested.")

    def predict_step(
        self, batch, batch_idx, dataloader_idx=0
    ) -> dict[str, torch.Tensor]:
        embeddings = batch["embeddings"]
        attention_mask = batch["attention_mask"]
        residue_index = batch["residue_index"]
        aatype = batch["aatype"]
        output = self.forward(
            s_s_0=embeddings,
            aatype=aatype,
            attention_mask=attention_mask,
            residue_index=residue_index,
        )
        return output
