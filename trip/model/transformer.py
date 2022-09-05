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

import logging
from typing import Optional, Literal, Dict

import torch
import torch.nn as nn
from dgl import DGLGraph
from torch import Tensor

from se3_transformer.model.basis import get_basis, update_basis_with_fused
from se3_transformer.model.layers.attention import AttentionBlockSE3
from se3_transformer.model.layers.convolution import ConvSE3, ConvSE3FuseLevel
from se3_transformer.model.layers.norm import NormSE3
from se3_transformer.model.layers.pooling import GPooling
from se3_transformer.runtime.utils import str2bool
from se3_transformer.model.fiber import Fiber

from dgl.nn import SumPooling
from se3_transformer.model.transformer import Sequential

from trip.model.norm import TrIPNorm
from trip.model.pooling import SumPoolingEdges
from trip.model.weighted_edge_softmax import WeightedEdgeSoftmax


def get_populated_edge_features(relative_pos: Tensor, edge_features: Optional[Dict[str, Tensor]] = None):
    """ Add relative positions to existing edge features """
    edge_features = edge_features.copy() if edge_features else {}
    r = relative_pos.norm(dim=-1, keepdim=True)
    r = torch.sqrt(r**2 + 1) - 1  # Smooth norm TODO: Optimize this with previous line
    if '0' in edge_features:
        edge_features['0'] = torch.cat([edge_features['0'], r[..., None]], dim=1)
    else:
        edge_features['0'] = r[..., None]

    return edge_features

class SE3TransformerTrIP(nn.Module):
    def __init__(self,
                 num_layers: int,
                 fiber_in: Fiber,
                 fiber_hidden: Fiber,
                 fiber_out: Fiber,
                 num_heads: int,
                 channels_div: int,
                 fiber_edge: Fiber = Fiber({}),
                 norm: bool = True,
                 use_layer_norm: bool = True,
                 tensor_cores: bool = False,
                 low_memory: bool = False,
                 **kwargs):
        """
        :param num_layers:          Number of attention layers
        :param fiber_in:            Input fiber description
        :param fiber_hidden:        Hidden fiber description
        :param fiber_out:           Output fiber description
        :param fiber_edge:          Input edge fiber description
        :param num_heads:           Number of attention heads
        :param channels_div:        Channels division before feeding to attention layer
        :param norm:                Apply a normalization layer after each attention block
        :param use_layer_norm:      Apply layer normalization between MLP layers
        :param tensor_cores:        True if using Tensor Cores (affects the use of fully fused convs, and padded bases)
        :param low_memory:          If True, will use slower ops that use less memory
        """
        super().__init__()
        self.num_layers = num_layers
        self.fiber_edge = fiber_edge
        self.num_heads = num_heads
        self.channels_div = channels_div
        self.max_degree = max(*fiber_in.degrees, *fiber_hidden.degrees, *fiber_out.degrees)
        self.tensor_cores = tensor_cores
        self.low_memory = low_memory

        if low_memory:
            self.fuse_level = ConvSE3FuseLevel.NONE
        else:
            # Fully fused convolutions when using Tensor Cores (and not low memory mode)
            self.fuse_level = ConvSE3FuseLevel.FULL if tensor_cores else ConvSE3FuseLevel.PARTIAL

        graph_modules = []
        for i in range(num_layers):
            graph_modules.append(AttentionBlockSE3(fiber_in=fiber_in,
                                                   fiber_out=fiber_hidden,
                                                   fiber_edge=fiber_edge,
                                                   num_heads=num_heads,
                                                   channels_div=channels_div,
                                                   use_layer_norm=use_layer_norm,
                                                   max_degree=self.max_degree,
                                                   fuse_level=self.fuse_level,
                                                   low_memory=low_memory,
                                                   edge_softmax_fn=WeightedEdgeSoftmax()))
            if norm:
                graph_modules.append(NormSE3(fiber_hidden))
            fiber_in = fiber_hidden

        graph_modules.append(ConvSE3(fiber_in=fiber_in,
                                     fiber_out=fiber_out,
                                     fiber_edge=fiber_edge,
                                     self_interaction=False,
                                     pool=False,
                                     use_layer_norm=use_layer_norm,
                                     max_degree=self.max_degree))
        self.graph_modules = Sequential(*graph_modules)

    def forward(self,
                graph: DGLGraph,
                node_feats: Dict[str, Tensor],
                edge_feats: Optional[Dict[str, Tensor]],
                scale: Tensor):
        # Compute bases in case they weren't precomputed as part of the data loading
        basis = get_basis(graph.edata['rel_pos'], max_degree=self.max_degree, compute_gradients=True,
                                   use_pad_trick=self.tensor_cores and not self.low_memory,
                                   amp=torch.is_autocast_enabled())

        # Add fused bases (per output degree, per input degree, and fully fused) to the dict
        basis = update_basis_with_fused(basis, self.max_degree, use_pad_trick=self.tensor_cores and not self.low_memory,
                                        fully_fused=self.fuse_level == ConvSE3FuseLevel.FULL)

        # Scale basis with cutoff function
        basis = {key: value * scale[...,None,None,None] for key, value in basis.items()}

        edge_feats = get_populated_edge_features(graph.edata['rel_pos'], edge_feats)

        feats = self.graph_modules(node_feats, edge_feats, graph=graph, basis=basis, scale=scale)
        return feats['0'].squeeze(-1)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--num_layers', type=int, default=7,
                            help='Number of stacked Transformer layers')
        parser.add_argument('--num_heads', type=int, default=8,
                            help='Number of heads in self-attention')
        parser.add_argument('--channels_div', type=int, default=2,
                            help='Channels division before feeding to attention layer')
        parser.add_argument('--norm', type=str2bool, nargs='?', const=True, default=False,
                            help='Apply a normalization layer after each attention block')
        parser.add_argument('--use_layer_norm', type=str2bool, nargs='?', const=True, default=False,
                            help='Apply layer normalization between MLP layers')
        parser.add_argument('--low_memory', type=str2bool, nargs='?', const=True, default=False,
                            help='If true, will use fused ops that are slower but that use less memory '
                                 '(expect 25 percent less memory). '
                                 'Only has an effect if AMP is enabled on Volta GPUs, or if running on Ampere GPUs')
        return parser

class TrIP(nn.Module):
    def __init__(self,
                 num_degrees: int,
                 num_channels: int,
                 cutoff: float,
                 **kwargs):
        super().__init__()
        self.num_degrees = num_degrees
        self.num_channels = num_channels
        self.cutoff = cutoff

        num_out_channels = num_channels * num_degrees
        self.transformer = SE3TransformerTrIP(
            fiber_in=Fiber.create(1, num_channels),
            fiber_hidden=Fiber.create(num_degrees, num_channels),
            fiber_out=Fiber.create(1, num_out_channels),
            fiber_edge=Fiber.create(1, self.num_channels - 1), # So there are num_channels when dist is cat'ed
            **kwargs
        )
        self.embedding = nn.Embedding(100, num_channels)
        self.mlp = nn.Sequential(
            nn.Linear(num_out_channels, num_out_channels),
            nn.SiLU(),
            nn.Linear(num_out_channels, 1)
        )
        self.pool = SumPoolingEdges()

    def forward(self, graph, forces=True, create_graph=True):
        scale = self.cutoff_fn(graph.edata['rel_pos'], self.cutoff)
        species_embedding = self.embedding(graph.ndata['species'] - 1)
        node_feats = {'0': species_embedding.unsqueeze(-1)}
        radial_basis = self._get_radial_basis(graph, self.cutoff, self.num_channels - 1)
        edge_feats = {'0': radial_basis.unsqueeze(-1)}

        feats = self.transformer(graph, node_feats, edge_feats, scale)
        atom_energies = self.mlp(feats).squeeze(-1)
        energies = self.pool(graph, atom_energies, weight=scale)
        if not forces:
            return energies
        forces = -torch.autograd.grad(torch.sum(energies),
                                      graph.ndata['pos'],
                                      create_graph=create_graph,
                                      )[0]
        return energies, forces
        
    @staticmethod
    def cutoff_fn(rel_pos, cutoff):
        dists = torch.norm(rel_pos, p=2, dim=1)
        scale = torch.zeros_like(dists)
        def bump_fn(x): return torch.exp(1 - 1 / (1 - x ** 2))  # Modified bump function with bump_fn(0) = 1
        def smooth_fn(x): return torch.exp(-1 / x)
        scale[dists < cutoff] = bump_fn(dists[dists < cutoff] / cutoff) * smooth_fn(dists[[dists < cutoff]])
        return scale

    @staticmethod
    def _get_radial_basis(graph, cutoff, num_basis_fns) -> Tensor:
        rel_pos = graph.edata['rel_pos']
        edge_dists = torch.norm(rel_pos, dim=1)
        gaussian_centers = torch.linspace(0, cutoff, num_basis_fns, device=rel_pos.device)
        dx = gaussian_centers[1] - gaussian_centers[0]
        diffs = edge_dists[:,None] - gaussian_centers[None,:]
        return torch.exp(-2 * diffs**2 / dx**2)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Model architecture")
        SE3TransformerTrIP.add_argparse_args(parser)
        parser.add_argument('--num_degrees',
                            help='Number of degrees to use. Hidden features will have types [0, ..., num_degrees - 1]',
                            type=int, default=3)
        parser.add_argument('--num_channels', help='Number of channels for the hidden features', type=int, default=32)
        return parent_parser
