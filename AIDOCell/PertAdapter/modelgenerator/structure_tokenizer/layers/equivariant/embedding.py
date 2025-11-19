from dataclasses import asdict

import numpy as np

import torch
import torch.nn as nn

from modelgenerator.structure_tokenizer.configs.models_configs import (
    EdgeEmbeddingConfig,
)
from modelgenerator.structure_tokenizer.layers.equivariant.irreps import IrrepTensor
from modelgenerator.structure_tokenizer.layers.equivariant.primitives import (
    EquivariantDropout,
)
from modelgenerator.structure_tokenizer.utils.shape_utils import left_gather
from modelgenerator.structure_tokenizer.utils.init_params import init_linear_
from modelgenerator.structure_tokenizer.utils.misc import cdist


class NodeTypeEmbedding(nn.Module):
    def __init__(
        self,
        n_node_type: int,
        n_chain_type: int,
        d: int,
        dropout: float = 0.0,
        mask_node_type: bool = False,
    ) -> None:
        """
        Initializes the NodeTypeEmbedding module.

        Args:
            n_node_type (int): Number of node types.
            n_chain_type (int): Number of chain types.
            d (int): Dimensionality of embeddings.
            dropout (float): Dropout rate.
            mask_node_type (bool): If True, masks all node types.
        """
        super().__init__()
        self.mask_node_type = mask_node_type
        self.mask_token_id = n_node_type - 1
        self.type2scalar = nn.Embedding(n_node_type, d)
        self.type2vec = nn.Embedding(n_node_type, 3 * d)
        self.chain2scalar = nn.Embedding(n_chain_type, d)
        self.dropout = EquivariantDropout(dropout)

    def reset_parameters(self):
        init_linear_(self.type2scalar, "xavier")
        init_linear_(self.type2vec, "xavier")
        init_linear_(self.chain2scalar, "xavier")

    def forward(
        self, node_type: torch.Tensor, rotmat: torch.Tensor, chain_id: torch.Tensor
    ) -> torch.Tensor:
        # extract input
        # shape: [*, N], [*, N, 3, 3], [*, N]
        if self.mask_node_type:
            node_type = torch.empty_like(node_type).fill_(self.mask_token_id)
        s = self.type2scalar(node_type) + self.chain2scalar(chain_id)  # [*, N, d]
        v = self.type2vec(node_type).reshape(*node_type.shape, -1, 3)  # [*, N, d, 3]
        node_irrep = IrrepTensor(s, v)
        node_irrep = node_irrep.rotate(rotmat)
        node_irrep = self.dropout(node_irrep)
        return node_irrep


class EdgeTypeEmbedding(nn.Module):
    def __init__(
        self, max_res_offset: int, max_sym_offset: int, d: int, dropout: float = 0.0
    ):
        """
        Initializes the EdgeTypeEmbedding module.

        Args:
            max_res_offset (int): Maximum residue offset.
            max_sym_offset (int): Maximum symmetry offset.
            d (int): Dimensionality of embeddings.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.max_res_offset = max_res_offset
        self.max_sym_offset = max_sym_offset
        self.res_offset_emb = nn.Embedding(
            2 * max_res_offset + 2, d
        )  # -max_res_offset, ..., 0, ..., max_res_offset, inf
        self.sym_offset_emb = nn.Embedding(
            2 * max_sym_offset + 2, d
        )  # -max_sym_offset, ..., 0, ..., max_sym_offset, inf
        self.dropout = nn.Dropout(dropout)
        init_linear_(self.res_offset_emb, "zero")
        init_linear_(self.sym_offset_emb, "zero")

    def forward(
        self,
        residue_index: torch.Tensor,
        chain_id: torch.Tensor,
        entity_id: torch.Tensor,
        sym_id: torch.Tensor,
        knn_key_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass to compute edge embeddings using residue and symmetry offsets.

        Args:
            residue_index (torch.Tensor): Residue offset tensor [N,].
            chain_id (torch.Tensor): Chain ID tensor [N,].
            entity_id (torch.Tensor): Entity ID tensor [N,].
            sym_id (torch.Tensor): Symmetry ID tensor [N,].
            knn_key_index (torch.Tensor, optional): KNN key index for gathering.

        Returns:
            torch.Tensor: Edge attributes after embedding.[Q, K, d]
        """
        rel_res_offset = self._compute_rel_res_offset(
            residue_index, chain_id, knn_key_index=knn_key_index
        )
        rel_sym_offset = self._compute_rel_sym_offset(
            sym_id, entity_id, knn_key_index=knn_key_index
        )
        emb = self.res_offset_emb(rel_res_offset) + self.sym_offset_emb(rel_sym_offset)
        emb = self.dropout(emb)
        return emb

    def _compute_rel_res_offset(
        self,
        residue_index: torch.Tensor,
        chain_id: torch.Tensor,
        knn_key_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # residue_index: [*, N]
        # chain_id: [*, N]
        same_chain = chain_id[..., None, :] == chain_id[..., None]  # [*, N, N]
        rel_res_offset = (
            residue_index[..., None, :] - residue_index[..., None]
        )  # [*, N, N]
        rel_res_offset = (
            (rel_res_offset + self.max_res_offset)
            .clamp(0, 2 * self.max_res_offset)
            .long()
        )
        rel_res_offset = torch.where(
            same_chain, rel_res_offset, 2 * self.max_res_offset + 1
        )
        if knn_key_index is not None:
            rel_res_offset = left_gather(
                rel_res_offset, -1, knn_key_index
            )  # [*, N, N] -> [*, N, k]
        return rel_res_offset

    def _compute_rel_sym_offset(
        self,
        sym_id: torch.Tensor,
        entity_id: torch.Tensor,
        knn_key_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        same_entity = entity_id[..., None, :] == entity_id[..., None]  # [*, N, N]
        rel_sym_offset = sym_id[..., None, :] - sym_id[..., None]  # [*, N, N]
        rel_sym_offset = (
            (rel_sym_offset + self.max_sym_offset)
            .clamp(0, 2 * self.max_sym_offset)
            .long()
        )
        rel_sym_offset = torch.where(
            same_entity, rel_sym_offset, 2 * self.max_sym_offset + 1
        )
        if knn_key_index is not None:
            rel_sym_offset = left_gather(rel_sym_offset, -1, knn_key_index)
        return rel_sym_offset


class BesselBasis(nn.Module):
    """
    A kind of sinusoidal embedding for length
    The Fourier–Bessel series may be thought of as a Fourier expansion in the ρ coordinate of cylindrical coordinates.
    See https://docs.e3nn.org/en/stable/api/math/math.html#e3nn.math.soft_one_hot_linspace for more details.
    """

    def __init__(self, bessel_const: float, d: int, eps: float = 1e-4):
        """
        Initializes the BesselBasis module.

        Args:
            bessel_const (float): Bessel constant for scaling.
            d (int): Dimensionality of the embeddings.
        """
        super().__init__()
        self.bessel_const = bessel_const
        self.eps = eps
        self.register_buffer("weights", torch.linspace(1, d, d) * np.pi)

    def forward(self, edge_len: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute sinusoidal embeddings.

        Args:
            edge_len (torch.Tensor): Edge length tensor [*,].

        Returns:
            torch.Tensor: Sinusoidal embeddings [*, d], with distance cutoff.
        """
        ret = (
            2
            / self.bessel_const
            * torch.sin(self.weights * edge_len[..., None] / self.bessel_const)
        )
        return ret * self.compute_cutoff(edge_len)

    @torch.no_grad()
    def compute_cutoff(self, edge_len: torch.Tensor) -> torch.Tensor:
        """
        prune out the long-range (> self.bessel_const) embeddings
        to understand the cutoff, see
        https://docs.e3nn.org/en/stable/api/math/math.html#e3nn.math.soft_one_hot_linspace
        """
        w = 10 * (1 - edge_len / self.bessel_const)
        w = torch.where(w > 0, (-1 / w.clip(min=self.eps)).exp(), 0)
        return w[..., None]  # [N, N, 1]


class MLP(nn.Module):
    def __init__(
        self,
        num_neurons: list[int],
        activation: nn.Module,
        apply_layer_norm: bool = False,
    ):
        super(MLP, self).__init__()
        self.activation = activation
        self.apply_layer_norm = apply_layer_norm
        layers = []
        for d_in, d_out in zip(num_neurons[:-1], num_neurons[1:]):
            linear = nn.Linear(d_in, d_out)
            init_linear_(linear, "xavier")
            layers.append(linear)
            if apply_layer_norm:
                layers.append(nn.LayerNorm(d_out))
            layers.append(activation())
        layers.pop()  # pop the last activation
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class EdgeEmb(nn.Module):
    def __init__(self, config: EdgeEmbeddingConfig):
        """
        Initializes the EdgeTypeEmbedding module.

        Args:
            max_res_offset (int): Maximum residue offset.
            max_sym_offset (int): Maximum symmetry offset.
            d (int): Dimensionality of embeddings.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.config = config
        self.eps = 1e-6
        self.edge_len_emb = BesselBasis(**asdict(self.config.bessels))
        self.edge_type_emb = EdgeTypeEmbedding(**asdict(config.edge_type_emb))
        self.mlp = MLP(
            num_neurons=[config.bessels.d + config.edge_type_emb.d, config.d, config.d],
            activation=nn.SiLU,
        )

    def _create_knn_key_index(
        self, trans: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor | None:
        """
        knn_key_index means the indices of the k nearest neighbors (including itself)
        and these indices are used to gather keys for attention
        """
        if self.config.k_for_knn is None:
            return None
        dist = cdist(
            trans, trans, mask_x=attention_mask, mask_y=attention_mask
        )  # [*, N, N]
        argsort = torch.argsort(dist, dim=-1)  # [*, N, N]
        knn_key_index = argsort[..., : self.config.k_for_knn]  # [*, N, k]
        return knn_key_index

    def forward(
        self,
        trans: torch.Tensor,
        attention_mask: torch.Tensor,
        residue_index: torch.Tensor,
        chain_id: torch.Tensor,
        entity_id: torch.Tensor,
        sym_id: torch.Tensor,
    ):
        edge_vec = trans[..., None, :, :] - trans[..., None, :]  # [*, N, N, 3]
        edge_len = torch.norm(edge_vec, dim=-1)  # [*, N, N]
        #
        irrep_scalar = torch.ones_like(edge_len)  # [*, N, N]
        irrep_vec = edge_vec / edge_len[..., None].clamp(min=self.eps)  # [*, N, N, 3]
        edge_irrep = IrrepTensor(irrep_scalar, irrep_vec)
        #
        pair_attn_mask = (
            attention_mask[..., None] * attention_mask[..., None, :]
        )  # [*, N, N]
        #
        knn_key_index = self._create_knn_key_index(
            trans=trans, attention_mask=attention_mask
        )
        if knn_key_index is not None:
            # apply knn_key_index to edge related tensors, changing the shape from [*, N, N] to [*, N, k]
            edge_len = left_gather(edge_len, -1, knn_key_index)  # [*, N, k]
            edge_irrep = edge_irrep.left_gather(-1, knn_key_index)  # [*, N, k]
            pair_attn_mask = left_gather(pair_attn_mask, -1, knn_key_index)  # [*, N, k]
        #
        edge_len_emb = self.edge_len_emb(edge_len)
        edge_type_emb = self.edge_type_emb(
            residue_index=residue_index,
            chain_id=chain_id,
            entity_id=entity_id,
            sym_id=sym_id,
            knn_key_index=knn_key_index,
        )
        edge_emb = torch.cat([edge_len_emb, edge_type_emb], dim=-1)
        edge_emb = self.mlp(edge_emb)  # [*, N, k, d] or [*, N, N, d]
        return {
            "edge_emb": edge_emb,
            "edge_irrep": edge_irrep,
            "pair_attn_mask": pair_attn_mask,
            "knn_key_index": knn_key_index,
        }
