from dataclasses import dataclass
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .utils.residual_vq import GroupedResidualVQ

from ..models import ConvNeXtBlock


@dataclass
class GRVQResult:
    quantized: torch.Tensor
    codes: torch.Tensor
    codes_list: list
    total_loss: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_diversity_loss: namedtuple
    quantized_fup: torch.Tensor
    quantized_fup_list: list
    x_pjt_in: torch.Tensor
    x_pjt_in_list: list


class DownsampleGRVQ(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 1,
        n_groups: int = 1,
        codebook_size: int = 1024,
        codebook_dim: int = 0,
        levels: list = [8, 5, 5, 5],
        downsample_factor: tuple[int] = (2, 2),
        downsample_dims: tuple[int] | None = None,
        ema_decay=0.8,
        codebook_diversity_loss_weight=0.0,
        codebook_diversity_temperature=100.0
    ):
        super().__init__()

        if downsample_dims is None:
            downsample_dims = [input_dim for i, _ in enumerate(range(len(downsample_factor)))]

        all_dims = (input_dim,) + tuple(downsample_dims)
        
        if codebook_dim == 0:
            codebook_dim = input_dim

        self.ema_decay = ema_decay
        self.codebook_diversity_loss_weight = codebook_diversity_loss_weight
        self.codebook_diversity_temperature = codebook_diversity_temperature
        self.grvq = GroupedResidualVQ(
            dim=all_dims[-1],
            num_quantizers=n_codebooks,
            groups=n_groups,
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,
            decay=self.ema_decay,
            codebook_diversity_loss_weight=self.codebook_diversity_loss_weight,
            codebook_diversity_temperature=self.codebook_diversity_temperature)

        self.downsample_factor = downsample_factor
        self.downsample_dims = downsample_dims

        self.downsample = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv1d(
                        all_dims[idx],
                        all_dims[idx + 1],
                        kernel_size=factor,
                        stride=factor,
                    ),
                    ConvNeXtBlock(dim=all_dims[idx + 1]),
                )
                for idx, factor in enumerate(downsample_factor)
            ]
        )

        self.upsample = nn.Sequential(
            *[
                nn.Sequential(
                    nn.ConvTranspose1d(
                        all_dims[idx + 1],
                        all_dims[idx],
                        kernel_size=factor,
                        stride=factor,
                    ),
                    ConvNeXtBlock(dim=all_dims[idx]),
                )
                for idx, factor in reversed(list(enumerate(downsample_factor)))
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, encoded_feature) -> GRVQResult:
        original_shape = encoded_feature.shape
        encoded_ds = self.downsample(encoded_feature)
        quantized_fdown, all_indices, total_losses, quantized_fup, x_pjt_in, loss_breakdown, loss_commit = self.grvq(encoded_ds.mT)
        quantized_ds = self.upsample(quantized_fdown.mT)
        result = GRVQResult(
            quantized=quantized_ds,
            codes=all_indices,
            codes_list=[],
            total_loss=torch.mean(total_losses),
            commitment_loss=torch.mean(loss_commit),
            codebook_diversity_loss=torch.mean(loss_breakdown),
            quantized_fup=quantized_fup,
            quantized_fup_list=[],
            x_pjt_in=x_pjt_in,
            x_pjt_in_list=[])

        # Pad or crop z to match original shape
        diff = original_shape[-1] - result.quantized.shape[-1]
        left = diff // 2
        right = diff - left

        if diff > 0:
            result.quantized = F.pad(result.quantized, (left, right))
        elif diff < 0:
            result.quantized = result.quantized[..., left:-right]

        return result

    def encode(self, encoded_feature):
        encoded_feature_ds = self.downsample(encoded_feature)
        _, indices, *_ = self.grvq(encoded_feature_ds.mT)
        indices = rearrange(indices, "g b l r -> b (g r) l")
        
        return indices

    def decode(self, indices: torch.Tensor):
        indices_rearrange = indices # rearrange(indices, "b (g r) l -> g b l r", g=self.grvq.groups)
        z_q = self.grvq.get_output_from_indices(indices_rearrange)
        z_q = self.upsample(z_q.mT)
        
        return z_q
    