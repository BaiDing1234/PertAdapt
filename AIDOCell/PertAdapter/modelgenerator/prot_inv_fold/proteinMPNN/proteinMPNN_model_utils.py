"""
Code in this file is adapted from 
- https://github.com/BytedProtein/ByProt/blob/main/src/byprot/models/fixedbb/__init__.py
- https://github.com/BytedProtein/ByProt/blob/main/src/byprot/models/fixedbb/generator.py
"""

import torch
from torch import nn
import numpy as np

from .proteinMPNN_data_utils import Alphabet

from .proteinMPNN_decoder import MPNNSequenceDecoder
from .proteinMPNN_encoder import MPNNEncoder


class FixedBackboneDesignEncoderDecoder(nn.Module):
    _default_cfg = {}

    def __init__(self, cfg) -> None:
        super().__init__()
        self._update_cfg(cfg)

    def _update_cfg(self, cfg):
        from omegaconf import OmegaConf

        self.cfg = OmegaConf.merge(self._default_cfg, cfg)

    @classmethod
    def from_config(cls, cfg):
        raise NotImplementedError

    def forward_encoder(self, batch):
        raise NotImplementedError

    def forward_decoder(self, prev_decoder_out, encoder_out):
        raise NotImplementedError

    def initialize_output_tokens(self, batch, encoder_out):
        raise NotImplementedError

    def forward(self, coords, coord_mask, tokens, token_padding_mask=None, **kwargs):
        raise NotImplementedError

    def sample(
        self, coords, coord_mask, tokens=None, token_padding_mask=None, **kwargs
    ):
        raise NotImplementedError


## Replaced for "from byprot.models.fixedbb.generator import new_arange, sample_from_categorical"
def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


def sample_from_categorical(logits=None, temperature=1.0):
    if temperature and False:
        dist = torch.distributions.Categorical(logits=logits.div(temperature))
        tokens = dist.sample()
        scores = dist.log_prob(tokens)
    else:
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores
