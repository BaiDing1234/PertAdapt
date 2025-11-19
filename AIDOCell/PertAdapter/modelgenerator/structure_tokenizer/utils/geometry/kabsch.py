import torch


def mask_average(x, mask=None, keepdim=False):
    """
    x: (*, L, D)
    mask: (*, L). mask = 1, if not masked. it can also be understood as weight.
    Return (*, D)
    """
    if mask is None:
        return x.mean(-2, keepdim=keepdim)
    mask = mask.unsqueeze(-1)
    Sum = (x * mask).sum(-2, keepdim=keepdim)
    return Sum / mask.sum(-2, keepdim=keepdim).clamp(min=1e-4)


def apply_mask(x, mask, pad_value=0):
    """
    x: (*, L, D)
    mask: (n_chain, L). mask = 1 if not masked.
    """
    if mask is None:
        return x
    mask = mask.unsqueeze(-1)
    if pad_value == 0:
        return x * mask
    else:
        return x * mask + (~mask) * pad_value


def find_rigid_alignment(A, B, mask=None):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Torch tensor of shape (*,N,D) -- Point Cloud to Align (source)
        -    B: Torch tensor of shape (*,N,D) -- Reference Point Cloud (target)

        Returns:
        -    R: optimal rotation
        -    t: optimal translation

        A @ R.T + t ~= B
    """
    with torch.amp.autocast(device_type="cuda", enabled=False):
        a_mean = mask_average(A, mask, keepdim=True)  # mean with keep dim. [*, 1, D]
        b_mean = mask_average(B, mask, keepdim=True)  # mean with keep dim. [*, 1, D]

        A_c = A - a_mean  # [*, N, D]
        B_c = B - b_mean  # [*, N, D]

        A_c = apply_mask(A_c, mask)  # [*, D, D]
        B_c = apply_mask(B_c, mask)  # [*, D, D]

        # Covariance matrix
        H = A_c.transpose(-1, -2) @ (B_c)  # # [*, D, D]
        U, S, Vh = torch.linalg.svd(H)  # [*, D, D]
        V = Vh.transpose(-1, -2)
        reflect = torch.det(U) * torch.det(V) < 0  # [*,]
        # flip negative
        # copy to avoid in-place op
        new_V = torch.zeros_like(V)
        new_V[..., :, :-1] = V[..., :, :-1]
        new_V[..., :, -1] = V[..., :, -1] * torch.where(reflect[..., None], -1, 1)
        V = new_V
        # Rotation matrix
        R = V @ U.transpose(-1, -2)
        # Translation vector
        t = b_mean - a_mean @ R.transpose(-1, -2)  # [*, 1, D]
        t = t.squeeze(-2)
        return R, t
