import torch
import numpy as np
from omegaconf import OmegaConf


def get_config_from_dict(config_dict: dict, config: type):
    schema = OmegaConf.structured(config)
    config = OmegaConf.create(config_dict)
    merged = OmegaConf.merge(schema, config)
    return OmegaConf.to_object(merged)


def cdist(
    x: torch.Tensor,
    y: torch.Tensor,
    mask_x: torch.Tensor | None = None,
    mask_y: torch.Tensor | None = None,
    zero_diag: bool = False,
) -> torch.Tensor:
    # where mask is False, the distance is set to inf
    cdist = torch.cdist(x, y)
    if zero_diag:
        assert (
            cdist.shape[-1] == cdist.shape[-2]
        ), f"Zeroing diagonal is only supported for square matrix, got {cdist.shape}"
        N = cdist.shape[-1]
        device = cdist.device
        eye = torch.eye(N, dtype=torch.bool, device=device)
        cdist = torch.where(eye, 0, cdist)
    if mask_x is not None:
        cdist = torch.where(mask_x[..., :, None].bool(), cdist, np.inf)
    if mask_y is not None:
        cdist = torch.where(mask_y[..., None, :].bool(), cdist, np.inf)
    return cdist
