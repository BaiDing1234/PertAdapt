from einops import rearrange

import torch
import torch.nn as nn

from modelgenerator.structure_tokenizer.layers.equivariant.primitives import (
    EquivariantLayerNorm,
    EquivariantLinear,
    EquivariantMultiHeadLinear,
    EquivariantDropout,
    depthwise_tensor_product,
    equivariant_gate_activation,
)
from modelgenerator.structure_tokenizer.utils.types import IrrepShape
from modelgenerator.structure_tokenizer.layers.equivariant.irreps import IrrepTensor
from modelgenerator.structure_tokenizer.utils.init_params import (
    init_params_recursively_,
)

LAYER_NORM_ON_SKIP = True


class GraphAttention(nn.Module):
    def __init__(
        self,
        d: int,
        n_head: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        add_bias: bool = True,
    ):
        super().__init__()
        self.d = d
        self.n_head = h = n_head
        self.add_bias = add_bias

        # dropouts
        self.attn_dropout = nn.Dropout(attn_dropout)  # dropout for attn
        self.dropout = EquivariantDropout(dropout)  # dropout for s and v

        # linear layers
        dh = d // h
        self.linear_query = EquivariantLinear(
            IrrepShape(d, d), IrrepShape(d, d), add_bias=add_bias
        )
        self.linear_key = EquivariantLinear(
            IrrepShape(d, d), IrrepShape(d, d), add_bias=add_bias
        )
        self.linear_query_key = EquivariantLinear(
            IrrepShape(2 * dh, 2 * dh), IrrepShape(dh, dh), add_bias=add_bias
        )
        self.linear_dtp_weights = nn.Linear(d, 4 * d, bias=add_bias)
        # scalar 2d/h -> 3d/h, vector 2d/h -> d/h
        # 3 of the scalars are attn, gate, value
        # 1 of the vectors is value
        self.linear_attn_gate_value = EquivariantMultiHeadLinear(
            h,
            IrrepShape(2 * dh, 2 * dh),
            IrrepShape(3 * dh, dh),
            add_bias=add_bias,
        )
        self.linear_attn = nn.Linear(dh, 1, bias=False)
        self.linear_value = EquivariantLinear(
            IrrepShape(2 * dh, 2 * dh), IrrepShape(dh, dh), add_bias=add_bias
        )
        self.linear_out = EquivariantLinear(
            IrrepShape(d, d), IrrepShape(d, d), add_bias=add_bias, init_type="zero"
        )

        # layer norms
        self.layer_norm_in = EquivariantLayerNorm(IrrepShape(d, d))
        self.layer_norm_attn = nn.LayerNorm(dh, dh)

    def reset_parameters(self):
        return init_params_recursively_(self)

    def _reserve_skip_connection(self, query):
        if LAYER_NORM_ON_SKIP:
            query = self.layer_norm_in(query)  # [*, Q, d]
            query0 = query.copy()  # shallow copy
        else:
            query0 = query.copy()  # shallow copy
            query = self.layer_norm_in(query)  # [*, Q, d]
        return query, query0

    def _query_key_projection(self, query, key):
        query = self.linear_query(query)  # [*, Q, d]
        key = self.linear_key(key)  # [*, K, d]
        return query, key

    def _qk_product(self, query, key, knn_key_index=None):
        # query: [*, Q, d]
        # key: [*, K, d]
        # knn_key_index: [*, Q, K'=KNN] is an index tensor telling which keys are in the KNN set

        # get shape info
        n_batch_dim = len(query.s.shape) - 2  # [*]
        Q = query.s.shape[-2]
        # add dims
        query = query.apply(lambda x: x.unsqueeze(n_batch_dim + 1))  # [*, Q, 1, d]
        key = key.apply(lambda x: x.unsqueeze(n_batch_dim))  # [*, 1, K, d]
        # remove non-knn keys
        if knn_key_index is not None:
            key = key.expand_one_dim(-3, Q)  # [*, Q, K, d]
            key = key.left_gather(-2, knn_key_index)  # [*, Q, K'=KNN, d]
        # query-key product
        qk = depthwise_tensor_product(query, key)  # [*, Q, K, 2d]
        # extract head dim to the front
        qk = qk.rearrange(
            "... Q K (h d) -> h ... Q K d", h=self.n_head
        )  # [h, *, Q, K, 2d/h]
        qk = self.linear_query_key(qk)  # [h, *, Q, K, d/h]
        return qk

    def _qk_edge_product(self, qk, edge_emb, edge_irrep):
        dtp_weights = self.linear_dtp_weights(edge_emb)  # [*, Q, K, 4d]
        dtp_weights = rearrange(
            dtp_weights, "... Q K (c h d) -> c h ... Q K d", c=4, h=self.n_head
        )  # [4, h, *, Q, K, d/h]
        edge_irrep = edge_irrep[
            ..., None
        ]  # [*, Q, K, 1], reserving the last dim for channel
        qk = depthwise_tensor_product(qk, edge_irrep, dtp_weights)  # [h, *, Q, K, 2d/h]
        return qk

    def _process_attn_feat(self, attn_feat, attn_mask=None):
        attn_feat = self.layer_norm_attn(attn_feat)  # [h, *, Q, K, d/h]
        # squeeze the last channel dim
        attn = self.linear_attn(attn_feat).squeeze(-1)  # [h, *, Q, K]
        if attn_mask is not None:
            attn = torch.where(
                attn_mask[None].bool(), attn, torch.finfo(attn.dtype).min
            )
        attn = attn.softmax(dim=-1)  # [h, *, Q, K]
        attn = self.attn_dropout(attn)
        return attn

    def _extract_attn_and_value(self, qk, attn_mask=None):
        # attn_gate_value_s, value_v: [h, *, Q, K, 3d/h], [h, *, Q, K, d/h, 3], scalar contains attn, gate, value
        attn_gate_value_s, value_v = self.linear_attn_gate_value(qk)
        # unbind attn_feat, gate, value_s: [h, *, Q, K, d/h]
        attn_feat, gate, value_s = rearrange(
            attn_gate_value_s, "h ... Q K (c d) -> c h ... Q K d", c=3
        )
        value = IrrepTensor(value_s, value_v)  # [h, *, Q, K, d/h]
        attn = self._process_attn_feat(attn_feat, attn_mask)  # [h, *, Q, K]
        value = equivariant_gate_activation(value, gate)  # [h, *, Q, K, d/h]
        return attn, value  # [h, *, Q, K], [h, *, Q, K, d/h]

    def _value_edge_product(self, value, edge_irrep):
        value = depthwise_tensor_product(
            value, edge_irrep[..., None]
        )  # [h, *, Q, K, 2d/h]
        value = self.linear_value(value)  # [h, *, Q, K, d/h]
        return value

    def _attn_value_product(self, attn, value):
        out = (value * attn[..., None]).sum(dim=-2)  # [h, *, Q, d/h]
        out = out.rearrange("h ... Q d -> ... Q (h d)", h=self.n_head)  # [*, Q, d]
        return out

    def forward(
        self,
        query: IrrepTensor,  # [*, Q, d]
        key: IrrepTensor,  # [*, K, d]
        edge_irrep: IrrepTensor,  # [*, Q, K]
        edge_emb: torch.Tensor,  # [*, Q, K, d]
        pair_attn_mask: torch.Tensor | None = None,  # [*, Q, K]
        knn_key_index: torch.Tensor | None = None,  # [*, Q, K'=KNN]
    ) -> tuple[IrrepTensor, torch.Tensor]:
        # reserve skip connection, including layer norm
        query, query0 = self._reserve_skip_connection(query)  # [*, Q, d]

        # similar to attention, but consider edges
        query, key = self._query_key_projection(query, key)  # [*, Q, d], [*, K, d]
        qk = self._qk_product(query, key, knn_key_index)  # [h, *, Q, K, d/h]
        qk = self._qk_edge_product(qk, edge_emb, edge_irrep)  # [h, *, Q, K, 2d/h]
        attn, value = self._extract_attn_and_value(
            qk, pair_attn_mask
        )  # [h, *, Q, K], [h, *, Q, K, d/h]
        value = self._value_edge_product(value, edge_irrep)  # [h, *, Q, K, d/h]
        out = self._attn_value_product(attn, value)  # [*, Q, d]
        out = self.linear_out(out)  # [*, Q, d]
        out = self.dropout(out)  # [*, Q, d]

        # skip connection
        out = out + query0
        return out, attn


class FFN(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_mult: int,
        dropout: float = 0.0,
        add_bias: bool = True,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_mult = d_mult
        self.layer_norm = EquivariantLayerNorm(IrrepShape(d_in, d_in))
        self.ff1 = EquivariantLinear(
            IrrepShape(d_in, d_in),
            IrrepShape(2 * d_mult * d_out, d_mult * d_out),
            add_bias=add_bias,
        )
        self.ff2 = EquivariantLinear(
            IrrepShape(d_mult * d_out, d_mult * d_out),
            IrrepShape(d_out, d_out),
            add_bias=add_bias,
            init_type="zero",
        )
        self.dropout = EquivariantDropout(dropout)

    def _reserve_skip_connection(self, x):
        if LAYER_NORM_ON_SKIP:
            x = self.layer_norm(x)
            x0 = x.copy()  # for skip connection
        else:
            x0 = x.copy()
            x = self.layer_norm(x)
        return x, x0

    def forward(self, x: IrrepTensor) -> IrrepTensor:
        x, x0 = self._reserve_skip_connection(x)  # [N, d_in]
        x = self.ff1(x)  # [N, 2*mul*d_out], [N, mul*d_out, 3], scalar contains gate
        x = equivariant_gate_activation(
            x
        )  # [N, 2*mul*d_out], [N, mul*d_out, 3] -> [N, mul*d_out], [N, mul*d_out, 3]
        x = self.ff2(x)  # [N, d_out]
        x = self.dropout(x)
        if self.d_out == self.d_in:
            x = x + x0
        return x


class Equiformer(nn.Module):
    """Implements Fig.1b of Equiformer, with several modifications"""

    def __init__(
        self,
        d: int,
        n_head: int,
        d_mult: int = 3,
        d_out: int | None = None,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.graph_attn = GraphAttention(d, n_head, dropout, attn_dropout)
        if d_out is None:
            d_out = d
        self.ffn = FFN(d, d_out, d_mult, dropout)

    def forward(
        self,
        node_irrep: IrrepTensor,
        edge_irrep: IrrepTensor,
        edge_emb: torch.Tensor,
        pair_attn_mask: torch.Tensor,
        knn_key_index: torch.Tensor | None = None,
    ) -> IrrepTensor:
        # shape: [N, d], [N, d, 3], [N, N], [N, N, 3], [N, N, d]
        out, attn = self.graph_attn(
            query=node_irrep,
            key=node_irrep,
            edge_irrep=edge_irrep,
            edge_emb=edge_emb,
            pair_attn_mask=pair_attn_mask,
            knn_key_index=knn_key_index,
        )
        out = self.ffn(out)
        return out
