# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

from typing import Dict

import torch
from torch import nn
from torch import Tensor
from torch.nn import init
from torch.cuda.nvtx import range as nvtx_range
from torch.nn.parameter import Parameter

from se3_transformer.model import Fiber



class TrIPNorm(nn.Module):
    """
    Norm-based SE(3)-equivariant nonlinearity.

                 ┌──> feature_norm ──> LayerNorm() ──> ReLU() ──┐
    feature_in ──┤                                              * ──> feature_out
                 └──> feature_phase ────────────────────────────┘
    """

    NORM_CLAMP = 2 ** -12  # Minimum positive subnormal for FP16  # TRIP

    def __init__(self, fiber: Fiber, nonlinearity: nn.Module = lambda x : x):
        super().__init__()
        self.fiber = fiber
        self.nonlinearity = nonlinearity

        if len(set(fiber.channels)) == 1:
            # Fuse all the layer normalizations into a group normalization
            self.group_norm = TrIPGroupNorm(fiber.channels[0], len(fiber.channels),)
        else:
        # Use multiple layer normalizations
            self.layer_norms = nn.ModuleDict({
                str(degree): TrIPLayerNorm(channels)
                for degree, channels in fiber
        })

    def forward(self, features: Dict[str, Tensor], *args, **kwargs) -> Dict[str, Tensor]:
        with nvtx_range('TrIPNorm'):
            output = {}
            if hasattr(self, 'group_norm'):
                # Compute per-degree norms of features
                norms = [features[str(d)].norm(dim=-1, keepdim=True).clamp(min=self.NORM_CLAMP)
                         for d in self.fiber.degrees]
                fused_norms = torch.cat(norms, dim=-1)
                new_norms = self.nonlinearity(self.group_norm(fused_norms))
                factor = new_norms / fused_norms
                for d in self.fiber.degrees:
                    output[str(d)] = features[str(d)] * factor[...,d].unsqueeze(-1)
            else:
                for degree, feat in features.items():
                    norm = feat.norm(dim=-1).clamp(min=self.NORM_CLAMP)
                    new_norm = self.nonlinearity(self.layer_norms[degree](norm))
                    output[degree] = feat * (new_norm / norm).unsqueeze(-1)
            return output

class TrIPLayerNorm(nn.Module):
    def __init__(self, num_channels, elementwise_affine=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TrIPLayerNorm, self).__init__()
        self.num_channels = num_channels
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(self.num_channels, **factory_kwargs))
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        out = input / torch.mean(input, dim=-1, keepdim=True)
        return out * self.weight if self.elementwise_affine else out
            
    def extra_repr(self) -> str:
        return '{normalized_shape}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


# TODO: Fix this code so it trains correctly
class TrIPGroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups, elementwise_affine=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TrIPGroupNorm, self).__init__()

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty((num_channels, num_groups), **factory_kwargs))
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        out = input / torch.mean(input, dim=-2, keepdim=True)
        return out * self.weight if self.elementwise_affine else out

    def extra_repr(self) -> str:
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={elementwise_affine}'.format(**self.__dict__)
