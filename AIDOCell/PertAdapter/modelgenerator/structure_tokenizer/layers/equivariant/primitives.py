"""
Primitive equivariant layers.
All the layers return IrrepTensor that contains scalars and vectors (up to l=1).
See Equiformer paper for more details about the layers.
"""

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelgenerator.structure_tokenizer.utils.types import IrrepShape
from modelgenerator.structure_tokenizer.layers.equivariant.irreps import IrrepTensor
from modelgenerator.structure_tokenizer.utils.init_params import init_linear_


def depthwise_tensor_product(
    x1: IrrepTensor,
    x2: IrrepTensor,
    weights: Union[None, torch.Tensor] = None,
) -> IrrepTensor:
    """
    depthwise tensor product between (s1, v1) and (s2, v2)
    Input:
        s1, s2: [..., d]
        v1, v2: [..., d, 3]
        weights: [4, ..., d] or None
    Return:
        s, v: [..., 2d], [..., 2d, 3]
    """
    ss = x1.s * x2.s
    sv = x1.s[..., None] * x2.v
    vs = x1.v * x2.s[..., None]
    vv = (x1.v * x2.v).sum(-1)
    # apply weights
    if weights is not None:
        w_ss, w_sv, w_vs, w_vv = weights.unbind(0)
        ss = ss * w_ss
        sv = sv * w_sv[..., None]
        vs = vs * w_vs[..., None]
        vv = vv * w_vv
    # concatenate
    s = torch.cat([ss, vv], -1)
    v = torch.cat([sv, vs], -2)
    return IrrepTensor(s, v)


def equivariant_gate_activation(
    x: IrrepTensor, gate: Union[None, torch.Tensor] = None
) -> IrrepTensor:
    s, v = x.s, x.v
    if gate is None:  # then gate is the last half of the s
        offset = s.shape[-1] // 2
        s, gate = s[..., :offset], s[..., offset:]
    s = F.silu(s)
    v = gate[..., None].sigmoid() * v
    return IrrepTensor(s, v)


class EquivariantLayerNorm(nn.Module):
    def __init__(self, d: IrrepShape, eps=1e-6):
        super().__init__()
        self.eps = eps

        self.gamma_s = nn.Parameter(torch.empty(d.s))
        self.beta_s = nn.Parameter(torch.empty(d.s))
        self.gamma_v = nn.Parameter(torch.empty(d.v))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.gamma_s)
        nn.init.zeros_(self.beta_s)
        nn.init.ones_(self.gamma_v)

    def forward(self, x: IrrepTensor) -> IrrepTensor:
        """
        Equivariant layer norm.
        Input:
            s: [*, d_s]
            v: [*, d_v, 3]
        Return:
            layer normed s: [*, d_s]
            layer normed v: [*, d_v, 3]
        """
        s, v = x.s, x.v
        # normalize s
        s = s - s.mean(dim=-1, keepdim=True)  # [*, d_s]
        rms_s = (s.square().mean(dim=-1, keepdim=True) + self.eps).sqrt()  # [*, 1]
        s = self.gamma_s * s / rms_s + self.beta_s
        # normalize v
        d_v = v.shape[-2]
        rms_v = (
            v.square().sum(dim=[-1, -2], keepdim=True) / d_v + self.eps
        ).sqrt()  # [*, 1, 1]
        v = self.gamma_v[:, None] * v / rms_v  # [*, d_v, 3]
        return IrrepTensor(s, v)


class EquivariantLinear(nn.Module):
    def __init__(
        self, d_in: IrrepShape, d_out: IrrepShape, add_bias=True, init_type="xavier"
    ):
        super().__init__()
        self.init_type = init_type

        self.linear_s = (
            nn.Linear(d_in.s, d_out.s, bias=add_bias) if d_out.s > 0 else None
        )
        self.linear_v = nn.Linear(d_in.v, d_out.v, bias=False) if d_out.v > 0 else None
        # bias must be False for vectors. only weights are used.

    def reset_parameters(self) -> None:
        if self.init_type == "xavier":
            init_linear_(self.linear_s, init_type="xavier")
            init_linear_(self.linear_v, init_type="xavier")
        elif self.init_type == "zero":
            init_linear_(self.linear_s, init_type="zero")
            init_linear_(self.linear_v, init_type="zero")
        else:
            raise ValueError(self.init_type)

    def forward(self, x: IrrepTensor) -> IrrepTensor:
        """
        Linear layer for mixing channels. Process scalars and vectors separately.
        Input:
            s: [*, d_in.s]
            v: [*, d_in.v, 3]
        Return:
            s: [*, d_out.s] or unchanged
            v: [*, d_out.v, 3] or unchanged
        """
        if self.linear_s is not None:
            s = self.linear_s(x.s)
        if self.linear_v is not None:
            v = torch.einsum("...ik,oi->...ok", x.v, self.linear_v.weight)
        return IrrepTensor(s, v)


class EquivariantDropout(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate

    def reset_parameters(self) -> None:
        pass

    def forward(self, x: IrrepTensor) -> IrrepTensor:
        """
        Dropout layer.
        Input:
            s: [*, d_s]
            v: [*, d_v, 3]
        Return:
            s: [*, d_s]
            v: [*, d_v, 3]
        """
        if not self.training or self.dropout_rate == 0:
            return x
        mask = torch.ones_like(x.s)
        mask = F.dropout(mask, p=self.dropout_rate, training=self.training)
        s = x.s * mask
        v = x.v * mask[..., None]
        return IrrepTensor(s, v)


class EquivariantMultiHeadLinear(nn.Module):
    def __init__(
        self,
        n_heads,
        d_in: IrrepShape,
        d_out: IrrepShape,
        add_bias=True,
        init_type="xavier",
    ):
        """
        Group linear layers by heads. No interaction betwen groups.
        """
        super().__init__()
        self.linears = nn.ModuleList()
        for _ in range(n_heads):
            self.linears.append(
                EquivariantLinear(d_in, d_out, add_bias=add_bias, init_type=init_type)
            )

    def reset_parameters(self) -> None:
        for linear in self.linears:
            linear.reset_parameters()

    def forward(self, x: IrrepTensor) -> IrrepTensor:
        """
        Input:
            s: [h, *, d_in.s]
            v: [h, *, d_in.v, 3]
        Return:
            s: [h, *, d_out.s]
            v: [h, *, d_out.v, 3]
        """
        per_head_s = []
        per_head_v = []
        # unbind according to heads
        for h, (ss, vv) in enumerate(zip(x.s.unbind(0), x.v.unbind(0))):
            ss, vv = self.linears[h](IrrepTensor(ss, vv))
            per_head_s.append(ss)
            per_head_v.append(vv)
        s = torch.stack(per_head_s, 0)
        v = torch.stack(per_head_v, 0)
        return IrrepTensor(s, v)
