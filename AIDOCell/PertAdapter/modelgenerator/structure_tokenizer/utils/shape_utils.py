import numpy as np
import torch
from typing import TypeVar, Sequence


TSequence = TypeVar("TSequence", bound=Sequence)


def expand_one_dim(tensor: torch.Tensor, dim: int, target_size: int):
    # expand the tensor along the specified dimension
    # e.g. expand(tensor, 1, 3) will expand the tensor from shape (a, 1, c) to (a, 3, c)
    expand_shape = list(tensor.shape)
    expand_shape[dim] = target_size
    return tensor.expand(*expand_shape)


def left_gather(tensor: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    # different from torch.gather, this function expands the rightmost dimension of index to match the tensor
    # e.g. tensor.shape = (a, b, c), index.shape = (a, d), dim = 1
    # left_gather(tensor, 1, index) will return a tensor of shape (a, d, c)
    while len(index.shape) < len(tensor.shape):
        index = index.unsqueeze(-1)
    expand_shape = list(tensor.shape)
    expand_shape[dim] = index.shape[dim]
    index = index.expand(*expand_shape)
    return tensor.gather(dim, index)


def stack_variable_length_tensors(
    sequences: Sequence[torch.Tensor],
    constant_value: int | float = 0,
    dtype: torch.dtype | None = None,
    return_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Automatically stack tensors together, padding variable lengths with the
    value in constant_value. Handles an arbitrary number of dimensions.

    Examples:
        >>> tensor1, tensor2 = torch.ones([2]), torch.ones([5])
        >>> stack_variable_length_tensors(tensor1, tensor2)
        tensor of shape [2, 5]. First row is [1, 1, 0, 0, 0]. Second row is all ones.

        >>> tensor1, tensor2 = torch.ones([2, 4]), torch.ones([5, 3])
        >>> stack_variable_length_tensors(tensor1, tensor2)
        tensor of shape [2, 5, 4]
    """
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype
    device = sequences[0].device

    array = torch.full(shape, constant_value, dtype=dtype, device=device)
    mask = torch.zeros(shape, dtype=torch.bool, device=device)

    for arr, msk, seq in zip(array, mask, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq
        msk[arrslice] = True

    if return_mask:
        return array, mask
    else:
        return array


def slice_python_object_as_numpy(
    obj: TSequence, idx: int | list[int] | slice | np.ndarray
) -> TSequence:
    """
    Slice a python object (like a list, string, or tuple) as if it was a numpy object.

    Example:
        >>> obj = "ABCDE"
        >>> slice_python_object_as_numpy(obj, [1, 3, 4])
        "BDE"

        >>> obj = [1, 2, 3, 4, 5]
        >>> slice_python_object_as_numpy(obj, np.arange(5) < 3)
        [1, 2, 3]
    """
    if isinstance(idx, int):
        idx = [idx]

    if isinstance(idx, np.ndarray) and idx.dtype == bool:
        sliced_obj = [obj[i] for i in np.where(idx)[0]]
    elif isinstance(idx, slice):
        sliced_obj = obj[idx]
    else:
        sliced_obj = [obj[i] for i in idx]

    match obj, sliced_obj:
        case str(), list():
            sliced_obj = "".join(sliced_obj)
        case _:
            sliced_obj = obj.__class__(sliced_obj)  # type: ignore

    return sliced_obj  # type: ignore
