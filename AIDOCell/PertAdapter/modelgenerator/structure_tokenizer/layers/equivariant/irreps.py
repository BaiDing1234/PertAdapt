from typing import Union, Tuple, List
from dataclasses import dataclass

import torch
from einops import rearrange

from modelgenerator.structure_tokenizer.utils.types import IrrepShape
from modelgenerator.structure_tokenizer.utils.shape_utils import (
    expand_one_dim,
    left_gather,
)


@dataclass
class IrrepTensor:
    """
    see https://docs.e3nn.org/en/stable/api/o3/o3_irreps.html
    """

    def __init__(self, s: torch.Tensor, v: torch.Tensor):
        self.s = s  # scalar, shape: [*]
        self.v = v  # vector, shape: [*, 3]
        # the prefix shape doesn't have to be the same,
        # but the last dim of v should be 3 and the shape of s should be 1 less than that of v

        assert len(s.shape) + 1 == len(
            v.shape
        ), f"The shape of s should be 1 less than that of v, got {s.shape} and {v.shape}"
        assert v.shape[-1] == 3, f"The last dim of v should be 3, got {v.shape[-1]}"

    def __iter__(self):
        # support `s, v = irreps`
        yield self.s
        yield self.v

    @property
    def shape(self) -> IrrepShape:
        return IrrepShape(self.s.shape, self.v.shape)

    def copy(self) -> "IrrepTensor":
        # shallow copy
        return IrrepTensor(self.s, self.v)

    def _ajust_v_dim(
        self, dim: Union[int, Tuple[int, ...], List[int]]
    ) -> Union[int, Tuple[int, ...], List[int]]:
        # adjust the dim for v used in sum, unsqueeze, etc., so that the last dim is always the 3-dim vector
        _ajust_v_dim_int = lambda d: d - 1 if d < 0 else d
        if isinstance(dim, int):
            return _ajust_v_dim_int(dim)
        elif isinstance(dim, tuple):
            return tuple([_ajust_v_dim_int(d) for d in dim])
        elif isinstance(dim, list):
            return [_ajust_v_dim_int(d) for d in dim]

    def __getitem__(self, idx: Union[int, slice, Tuple[int, ...]]) -> "IrrepTensor":
        # ref: pytorch tensor's __getitem__
        # https://github.com/pytorch/pytorch/blob/bbddde311a67797ce8409baf8f20e18757323261/functorch/dim/reference.py#L399
        assert isinstance(
            idx, (int, slice, tuple)
        ), f"Only simple indexing is supported, got {type(idx)}"

        if not isinstance(idx, tuple):
            idx = [idx]
        else:
            idx = list(idx)

        s_idx = idx
        v_idx = idx + [slice(None)]  # the last dim is always the 3-dim vector
        return IrrepTensor(self.s[s_idx], self.v[v_idx])

    def unsqueeze(self, dim: int) -> "IrrepTensor":
        IrrepTensor(self.s.unsqueeze(dim), self.v.unsqueeze(self._ajust_v_dim(dim)))

    def rearrange(self, pattern: str, **axes_lengths) -> "IrrepTensor":
        # see .venv/lib/python3.10/site-packages/einops/einops.py
        assert "d3" not in pattern.split(
            " "
        ), f"Please don't use d3 in the pattern, got {pattern}"
        s_pattern = pattern
        v_pattern_left, v_pattern_right = pattern.split("->")
        v_pattern = v_pattern_left + " d3 -> " + v_pattern_right + " d3"  # the
        return IrrepTensor(
            rearrange(self.s, s_pattern, **axes_lengths),
            rearrange(self.v, v_pattern, **axes_lengths, d3=3),
        )

    def apply(self, func):
        return IrrepTensor(func(self.s), func(self.v))

    def to(self, device: Union[int, str, torch.device]) -> "IrrepTensor":
        return self.apply(lambda x: x.to(device))

    def __add__(self, other: "IrrepTensor") -> "IrrepTensor":
        assert isinstance(
            other, IrrepTensor
        ), f"Only IrrepTensor can be added, got {type(other)}"
        return IrrepTensor(self.s + other.s, self.v + other.v)

    def __mul__(self, other: Union[torch.Tensor, float, int]) -> "IrrepTensor":
        assert isinstance(
            other, (torch.Tensor, float, int)
        ), f"Only support scalar mult. Only torch.Tensor, float, or int can be multiplied with IrrepTensor, got {type(other)}"

        if isinstance(other, torch.Tensor):
            return IrrepTensor(
                self.s * other, self.v * other[..., None]
            )  # the last dim is always the 3-dim vector
        elif isinstance(other, (float, int)):
            return IrrepTensor(self.s * other, self.v * other)
        else:
            raise ValueError(f"Unsupported type {type(other)}")

    def __rmul__(self, other: Union[torch.Tensor, float, int]) -> "IrrepTensor":
        return self.__mul__(other)

    def sum(
        self, dim: Union[int, Tuple[int, ...], List[int]], keepdim=False
    ) -> "IrrepTensor":
        return IrrepTensor(
            self.s.sum(dim, keepdim=keepdim),
            self.v.sum(self._ajust_v_dim(dim), keepdim=keepdim),
        )

    def rotate(self, rotmat: torch.Tensor) -> "IrrepTensor":
        # rotate the 3D vectors. Column-vector convention.
        assert rotmat.shape[-2:] == (
            3,
            3,
        ), f"Rotation matrix should be of shape [*, 3, 3], got {rotmat.shape}"
        return IrrepTensor(
            self.s, torch.einsum("...ij,...dj->...di", rotmat, self.v)
        )  # d is the channel dim

    def expand_one_dim(self, dim: int, target_size: int) -> "IrrepTensor":
        assert isinstance(dim, int), f"dim should be an integer, got {type(dim)}"
        assert isinstance(
            target_size, int
        ), f"target_size should be an integer, got {type(target_size)}"
        return IrrepTensor(
            expand_one_dim(self.s, dim, target_size),
            expand_one_dim(self.v, self._ajust_v_dim(dim), target_size),
        )

    def left_gather(self, dim: int, index: torch.Tensor) -> "IrrepTensor":
        assert isinstance(dim, int), f"dim should be an integer, got {type(dim)}"
        assert isinstance(
            index, torch.Tensor
        ), f"index should be a torch.Tensor, got {type(index)}"
        return IrrepTensor(
            left_gather(self.s, dim, index),
            left_gather(self.v, self._ajust_v_dim(dim), index),
        )
