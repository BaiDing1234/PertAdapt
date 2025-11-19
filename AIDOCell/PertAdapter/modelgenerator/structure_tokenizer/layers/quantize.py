# https://github.com/rosinality/vq-vae-2-pytorch/blob/ef5f67c46f93624163776caec9e0d95063910eca/vqvae.py#L27
from dataclasses import asdict
from functools import cached_property

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelgenerator.structure_tokenizer.configs.models_configs import QuantizeConfig
from modelgenerator.structure_tokenizer.utils.distributed import (
    all_reduce,
    get_world_size,
)

# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class Quantize(nn.Module):
    def __init__(
        self,
        dim: int,
        n_embed: int,
        normalize: bool = True,
        decay: float = 0.99,
        eps: float = 1e-5,
        usage_threshold: float = 1e-9,
        restart: int = 100,
    ):
        """
        Args:
            dim: int, dimension of the codebook
            n_embed: int, number of embeddings in the codebook
            normalize: bool, whether to normalize the codebook
            decay: float, decay for the EMA update of the codebook
            eps: float, small value to avoid numerical instability
            usage_threshold: float, codes below threshold will be reset to a random code

        normalize:
        borrowed from https://github.com/microsoft/unilm/blob/1c957d6ee7912196d59b02ce6caa60dbfe7e8937/beit2/norm_ema_quantizer.py
        """
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.normalize = normalize
        self.decay = decay
        self.eps = eps
        embed = torch.randn(dim, n_embed, requires_grad=False)
        if self.normalize:
            embed = F.normalize(embed, p=2, dim=0)
        self.register_buffer("embed", embed)  # [d, nb_codebooks]
        self.register_buffer("cluster_size", torch.zeros(n_embed, requires_grad=False))
        if not self.normalize:
            self.register_buffer("embed_avg", embed.clone())

        # https://www.reddit.com/r/MachineLearning/comments/nxjqvb/comment/h1g9dip
        # https://gist.github.com/a-kore/4befe292249098854f088c0c03606eda
        self.register_buffer("usage", torch.ones(self.n_embed, requires_grad=False))
        self.usage_threshold = usage_threshold
        self.restart = restart
        self.dead_codes = []
        self.frozen = False

    @cached_property
    def max_perplexity(self) -> torch.Tensor:
        """
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + self.eps)))
        max perplexity is when prob is uniform

        Returns:
            max perplexity of the codebook
        """
        prob = torch.ones(self.n_embed) / self.n_embed
        return (-prob * prob.clamp_min(self.eps).log()).sum().exp()

    @torch.no_grad()
    def update_usage(self, is_used: torch.Tensor):
        """
        Update the count of use  for each code in the codebook, and decay the usage of all codes.
        Enables a EMA update of the codebook usage.
        if code is used add 1 to usage then decay all codes usage by half.
        """

        self.usage[is_used] = self.usage[is_used] + 1
        self.usage = self.usage / 2

    @torch.no_grad()
    def reset_usage(self):
        """
        Reset the usage of all codes in the codebook to zero.
        """
        self.usage.zero_()

    @torch.no_grad()
    def random_restart(self):
        """
        randomly restart all dead codes below threshold with random code in codebook
        do this every several steps.
        self.usage_threshold parametrized the threshold below which a codebook is considered dead.
        """

        dead_codes = torch.nonzero(self.usage < self.usage_threshold).squeeze(1)
        self.dead_codes = dead_codes.tolist()
        alive_codes = torch.nonzero(self.usage >= self.usage_threshold).squeeze(1)
        choose_ids = torch.randint(0, alive_codes.shape[0], (dead_codes.shape[0],))
        rand_codes = alive_codes[choose_ids]
        # reset dead codes to random codes
        self.embed[:, dead_codes] = self.embed[:, rand_codes]
        all_reduce(self.embed)
        self.embed = self.embed / get_world_size()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Forward pass for quantization.
        Args:
            x: [*, d], input tensor to quantize, usually the output of the encoder

        Returns:
            quantize (torch.Tensor): [N, *], quantized tensor
            diff: (float), L1 distance between input and quantized tensor
            embed_ind: [*], index of the quantized tensor in the codebook
            prob: (torch.Tensor) [n_embed], probability of each code being used
            perplexity: float, perplexity of the codebook
        """
        assert (
            x.shape[-1] == self.dim
        ), f"Input of the Quantize module is of shape {x.shape} and {x.shape[-1]} is not equal to {self.dim}."
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        flatten = x.reshape(-1, self.dim)  # [N, dim]
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )  # [N, n_embed]
        _, embed_ind = (-dist).max(1)  # [N,]
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(
            flatten.dtype
        )  # [N, n_embed]
        embed_ind = embed_ind.view(*x.shape[:-1])  # [*]
        quantize = self.embed_code(embed_ind)  # [*, dim]

        if self.training and torch.is_grad_enabled() and not self.frozen:
            self._update_embeddings(flatten, embed_onehot)

        embed_onehot_sum = embed_onehot.sum(0)  # cluster size, [n_embed,]
        prob = embed_onehot_sum / (embed_onehot_sum.sum() + self.eps)  # [n_embed]
        perplexity = (-prob * prob.clamp_min(self.eps).log()).sum().exp()

        diff = (quantize.detach() - x).abs().mean()

        # Straight Through Step
        quantize = x + (quantize - x).detach()

        return quantize, diff, embed_ind, prob, perplexity

    def embed_code(self, embed_id):
        """
        Embeds the given indices using the codebook.
        Args:
            embed_id (torch.Tensor): Indices to embed.
        Returns:
            torch.Tensor: Embedded values.
        """
        return F.embedding(embed_id, self.embed.transpose(0, 1))

    @torch.no_grad()
    def _update_embeddings(self, flatten: torch.Tensor, embed_onehot: torch.Tensor):
        """
        Helper function to update embeddings during training.
        Args:
            flatten (torch.Tensor): Flattened input tensor.
            embed_onehot (torch.Tensor): One-hot encoded indices of embeddings.
        """
        embed_onehot_sum = embed_onehot.sum(0)  # cluster size, [n_embed,]
        embed_sum = flatten.T @ embed_onehot  # [dim, n_embed]

        all_reduce(embed_onehot_sum)
        all_reduce(embed_sum)

        is_used = (
            embed_onehot_sum > 0
        )  # it's not very accurate when batch size is small. you can decrease usage_threshold.
        self.update_usage(is_used)

        self.cluster_size.data.mul_(self.decay).add_(
            embed_onehot_sum, alpha=1 - self.decay
        )

        if self.normalize:
            # https://github.com/microsoft/unilm/blob/1c957d6ee7912196d59b02ce6caa60dbfe7e8937/beit2/norm_ema_quantizer.py
            embed_onehot_sum = embed_onehot_sum.masked_fill(~is_used, 1.0)  # [n_embed]
            embed_normalized = embed_sum / embed_onehot_sum[None]  # [dim, n_embed]
            embed_normalized = F.normalize(
                embed_normalized, p=2, dim=0
            )  # [dim, n_embed]
            embed_normalized = torch.where(
                is_used, embed_normalized, self.embed
            )  # [dim, n_embed]
            self.embed.data.mul_(self.decay).add_(
                embed_normalized, alpha=1 - self.decay
            )  # [dim, n_embed]
            self.embed.data = F.normalize(self.embed, p=2, dim=0)
        else:
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

    @classmethod
    def from_config(cls, config: QuantizeConfig) -> "Quantize":
        return cls(asdict(config))
