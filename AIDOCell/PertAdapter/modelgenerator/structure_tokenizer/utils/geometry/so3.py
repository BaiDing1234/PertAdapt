import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _rotmat_from_gram_schmidt(
    v1: torch.Tensor, v2: torch.Tensor, noise_eps: float = 0.0
) -> torch.Tensor:
    """
    After being orthogonalized and normalized, v1 and v2 will become the first two columns of the rotation matrix.

    Args:
        v1 (torch.Tensor): The first vector.
        v2 (torch.Tensor): The second vector.
        noise_eps (float): Noise epsilon to increase stability.

    Returns:
        torch.Tensor: The rotation matrix.
    """
    v1, v2 = (
        v1 + torch.randn_like(v1) * noise_eps,
        v2 + torch.randn_like(v2) * noise_eps,
    )  # add noise to increase stability
    # schmidt orth proj
    u1 = F.normalize(v1, dim=-1)
    u2 = F.normalize(v2 - (v2 * u1).sum(dim=-1, keepdim=True) * u1, dim=-1)
    u3 = torch.cross(
        u1, u2, dim=-1
    )  # use cross product to guanrantee u3 is orthogonal to u1 and u2
    rotmat = torch.stack([u1, u2, u3], dim=-1)  # [*, 3, 3]
    return rotmat


def _sanity_check_rotmat(
    rotmat: torch.Tensor, mask: torch.Tensor | None = None
) -> None:
    # check if the rotmat is orthogonal
    if mask is None:
        maxerr = (rotmat.det() - 1).abs().max()
    else:
        maxerr = (rotmat[mask != 0].det() - 1).abs().max()
    if not maxerr < 1e-5:
        logger.error(
            f"_rotmat_from_gram_schmidt returns non-orthogonal matrix: max-error={maxerr}. "
            "Please check the input vectors."
        )


def rotmat_from_gram_schmidt(
    v1: torch.Tensor,
    v2: torch.Tensor,
    mask: torch.Tensor | None = None,
    noise_eps: float = 0.0,
) -> torch.Tensor:
    """
    Compute a rotation matrix, where the first column is v1 and the second column is orthogonalized v2.
    v1, v2: [*, 3], two vectors to be orthonormalized
    mask: [*], 0 for invalid, 1 for valid
    Return: rotmat: [*, 3, 3]
    """
    rotmat = _rotmat_from_gram_schmidt(v1, v2, noise_eps=noise_eps)  # [*, 3, 3]
    ident = torch.eye(3, device=v1.device, dtype=v1.dtype)
    rotmat = torch.where((mask != 0)[..., None, None], rotmat, ident)  # [*, 3, 3]
    _sanity_check_rotmat(rotmat, mask=mask)
    return rotmat
