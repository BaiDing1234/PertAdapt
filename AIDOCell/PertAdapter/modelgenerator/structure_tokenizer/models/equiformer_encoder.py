from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from lightning.pytorch.utilities.types import STEP_OUTPUT

from modelgenerator.structure_tokenizer.configs.models_configs import (
    EquiformerEncoderConfig,
)
from modelgenerator.structure_tokenizer.layers.equivariant.equiformer import Equiformer
from modelgenerator.structure_tokenizer.layers.equivariant.embedding import (
    EdgeEmb,
    NodeTypeEmbedding,
)
from modelgenerator.structure_tokenizer.layers.quantize import Quantize
import lightning as L

from modelgenerator.structure_tokenizer.utils.constants.structure_tokenizer import (
    SCALE_POSITIONS,
    QUANTIZE_IDX_MASK,
)
from modelgenerator.structure_tokenizer.utils.init_params import (
    init_params_recursively_,
)

from modelgenerator.structure_tokenizer.utils.geometry.so3 import (
    rotmat_from_gram_schmidt,
)
from modelgenerator.structure_tokenizer.utils.misc import get_config_from_dict


class EquiformerEncoder(
    nn.Module,
    PyTorchModelHubMixin,
    # library_name="",
    # repo_url="",
    # docs_url="",
    # license="",
    tags=["biology", "genbio", "protein"],
):
    def __init__(self, config: EquiformerEncoderConfig):
        super().__init__()
        self.config = get_config_from_dict(
            config_dict=asdict(config), config=EquiformerEncoderConfig
        )
        self.node_emb: NodeTypeEmbedding = NodeTypeEmbedding(
            **asdict(self.config.node_emb)
        )
        self.edge_emb: EdgeEmb = EdgeEmb(self.config.edge_emb)
        self.eqnets = nn.ModuleList(
            [
                Equiformer(**asdict(self.config.eqnet))
                for _ in range(self.config.n_eqnet)
            ]
        )
        self.linear_in = nn.Linear(self.config.eqnet.d, self.config.quantize.dim)
        self.quantize = Quantize(**asdict(self.config.quantize))

    def reset_parameters(self):
        init_params_recursively_(self)

    def coords_to_rotmat_trans(
        self, atom_positions: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        atom_positions: [num_res, num_atom_type, 3]
        rotmat: [num_res, 3, 3]
        trans: [num_res, 3]

        Should we do this here / preprocessing ?
        """
        n, ca, c = atom_positions[..., :3, :].unbind(-2)
        trans = n / SCALE_POSITIONS
        rotmat = rotmat_from_gram_schmidt(ca - n, c - ca, mask=mask)
        return rotmat, trans

    def quantize_with_mask(
        self, emb: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        qx, diff_loss, idx, prob, ppl = self.quantize(emb[mask.bool()])
        qx_ = torch.zeros_like(emb)
        qx_[mask.bool()] = qx
        idx_ = (
            torch.zeros_like(mask, dtype=torch.long) - QUANTIZE_IDX_MASK
        )  # - QUANTIZE_IDX_MASK is a special value that will be ignored in the loss function
        idx_[mask.bool()] = idx
        return qx_, diff_loss, idx_, prob, ppl

    def forward(
        self,
        atom_positions: torch.Tensor,
        attention_mask: torch.Tensor,
        residue_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        rotmat, trans = self.coords_to_rotmat_trans(
            atom_positions=atom_positions, mask=attention_mask
        )
        node_irrep = self.node_emb(
            node_type=torch.zeros_like(residue_index),
            rotmat=rotmat,
            chain_id=torch.zeros_like(residue_index),
        )
        edge_embeddings = self.edge_emb(
            trans=trans,
            attention_mask=attention_mask,
            residue_index=residue_index,
            chain_id=torch.zeros_like(residue_index),
            entity_id=torch.zeros_like(residue_index),
            sym_id=torch.zeros_like(residue_index),
        )
        edge_emb = edge_embeddings["edge_emb"]
        edge_irrep = edge_embeddings["edge_irrep"]
        pair_attn_mask = edge_embeddings["pair_attn_mask"]
        knn_key_index = edge_embeddings["knn_key_index"]

        for layer_idx in range(self.config.n_eqnet):
            node_irrep = self.eqnets[layer_idx](
                node_irrep=node_irrep,
                edge_irrep=edge_irrep,
                edge_emb=edge_emb,
                pair_attn_mask=pair_attn_mask,
                knn_key_index=knn_key_index,
            )

        emb = node_irrep.s
        emb = F.normalize(emb, p=2, dim=-1)
        emb = self.linear_in(emb)
        qx, commitment_loss, idx, prob, ppl = self.quantize_with_mask(
            emb=emb, mask=attention_mask
        )
        logits = F.normalize(qx, p=2, dim=-1) @ self.quantize.embed  # [*, V]
        raw_qx = qx
        # Additional possible statistics
        # "codebook": self.codebook,
        return {
            "emb": emb,
            "raw_qx": raw_qx,
            "commitment_loss": commitment_loss,
            "logits": logits,
            "idx": idx,
            "prob": prob,
            "ppl": ppl,
            "dead_codes": torch.tensor(
                len(self.quantize.dead_codes), device=emb.device
            ),
        }

    def freeze_unused_params(self) -> list[str]:
        return self.freeze_last_layer_vec_heads()

    def freeze_last_layer_vec_heads(self) -> list[str]:
        # the last layer's vector outputs are not used in the loss computation
        frozen = []
        last_eqnet = self.eqnets[-1]
        idx = len(self.eqnets) - 1
        for name, params in last_eqnet.named_parameters():
            if "graph_attn.linear_value.linear_v" in name:
                params.requires_grad_(False)
            elif "graph_attn.linear_out.linear_v" in name:
                params.requires_grad_(False)
            elif "ffn.layer_norm.gamma_v" in name:
                params.requires_grad_(False)
            elif "ffn.ff1.linear_v" in name:
                params.requires_grad_(False)
            elif "ffn.ff2.linear_v" in name:
                params.requires_grad_(False)
            else:
                continue
            frozen.append("eqnets.{}.{}".format(idx, name))
        return frozen

    def restart(self) -> None:
        self.quantize.random_restart()

    @property
    def codebook(self) -> torch.Tensor:
        return self.quantize.embed.T.detach()


class EquiformerEncoderLightning(L.LightningModule):
    def __init__(self, pretrained_model_name_or_path: str) -> None:
        super().__init__()
        self.encoder = EquiformerEncoder.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        self.encoder.quantize.frozen = True

    def forward(
        self,
        atom_positions: torch.Tensor,
        attention_mask: torch.Tensor,
        residue_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        assert not self.training, "Only inference mode is supported."
        return self.encoder(
            atom_positions=atom_positions,
            attention_mask=attention_mask,
            residue_index=residue_index,
        )

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        raise NotImplementedError("EquiformerEncoderLightning can not be trained.")

    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        raise NotImplementedError("EquiformerEncoderLightning can not be validated.")

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        raise NotImplementedError("EquiformerEncoderLightning can not be tested.")

    def predict_step(
        self, batch, batch_idx, dataloader_idx=0
    ) -> dict[str, dict[str, torch.Tensor]]:
        atom_positions = batch["atom_positions"]
        attention_mask = batch["attention_mask"]
        residue_index = batch["residue_index"]
        aatype = batch["aatype"]
        batch_ids = batch["id"]
        batch_entity_ids = batch["entity_id"]
        batch_chain_ids = batch["chain_id"]
        output = self.forward(
            atom_positions=atom_positions,
            attention_mask=attention_mask,
            residue_index=residue_index,
        )
        min_encoding_indices = output["idx"]
        keys = [
            (
                f"{str(id)}_{str(entity_id)}_{str(chain_id)}"
                if entity_id is not None
                else f"{str(id)}_{str(chain_id)}"
            )
            for id, entity_id, chain_id in zip(
                batch_ids, batch_entity_ids, batch_chain_ids
            )
        ]
        struct_tokens = {
            k: {
                "struct_tokens": min_indices[mask],
                "aatype": aa[mask],
                "residue_index": res[mask],
            }
            for k, min_indices, aa, res, mask in zip(
                keys, min_encoding_indices, aatype, residue_index, attention_mask
            )
        }
        return struct_tokens
