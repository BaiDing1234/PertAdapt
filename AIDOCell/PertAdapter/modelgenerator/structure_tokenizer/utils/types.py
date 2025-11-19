from pathlib import Path
from dataclasses import dataclass

import torch

ShapeLike = int | torch.Size
PathLike = str | Path


# for what is irreps, see https://docs.e3nn.org/en/stable/api/o3/o3_irreps.html
# here you can just think of it as a tuple of scalars and vectors
@dataclass
class IrrepShape:
    s: ShapeLike
    v: ShapeLike
