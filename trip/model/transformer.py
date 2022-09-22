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

from typing import Optional, Literal, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dgl import DGLGraph

from se3_transformer.model.basis import get_basis, update_basis_with_fused
from se3_transformer.model.layers.attention import AttentionBlockSE3
from se3_transformer.model.layers.convolution import ConvSE3, ConvSE3FuseLevel
from se3_transformer.model.fiber import Fiber
from se3_transformer.model.transformer import Sequential, get_populated_edge_features
from se3_transformer.runtime.utils import str2bool

from trip.model.gate import TrIPGate
from trip.model.norm import TrIPNorm
from trip.model.pooling import SumPoolingEdges
from trip.model.weighted_edge_softmax import WeightedEdgeSoftmax

        
class SE3TransformerTrIP(nn.Module):
    def __init__(self,
                 num_layers: int,
                 fiber_in: Fiber,
                 fiber_hidden: Fiber,
                 fiber_out: Fiber,
                 num_heads: int,
                 channels_div: int,
                 fiber_edge: Fiber = Fiber({}),
                 gate: bool = True,
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

        fiber_conv = TrIPGate.make_input_fiber(fiber_hidden) if gate else fiber_hidden

        graph_modules = []
        for _ in range(num_layers):
            # TODO: Make fiber_conv and fiber_hidden distinction understandable
            graph_modules.append(AttentionBlockSE3(fiber_in=fiber_in,
                                                   fiber_out=fiber_conv,
                                                   fiber_edge=fiber_edge,
                                                   num_heads=num_heads,
                                                   channels_div=channels_div,
                                                   use_layer_norm=use_layer_norm,
                                                   max_degree=self.max_degree,
                                                   fuse_level=self.fuse_level,
                                                   low_memory=low_memory,
                                                   edge_softmax_fn=WeightedEdgeSoftmax()))
            if norm:
                graph_modules.append(TrIPNorm(fiber_conv, nonlinearity = lambda x : x))
            if gate:
                graph_modules.append(TrIPGate(fiber_conv))
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

        # Scale basis
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
        parser.add_argument('--gate', type=str2bool, nargs='?', const=True, default=False,
                            help='Apply a gated nonlinear layer after each attention block')
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
                 energy_std: float,
                 screen: float,
                 cutoff: float,
                 coulomb_energy_unit: float = 0.529_177_211,  # e^2/(4*pi*e0) in Ha*A
                 **kwargs):
        super().__init__()
        self.num_degrees = num_degrees
        self.num_channels = num_channels
        self.coulomb_energy_unit = coulomb_energy_unit
        self.screen = screen
        self.cutoff = cutoff
        self.energy_std = energy_std

        self.embedding = nn.Embedding(100, num_channels)
        num_out_channels = num_channels * num_degrees
        self.transformer = SE3TransformerTrIP(
            fiber_in=Fiber.create(1, num_channels),
            fiber_hidden=Fiber.create(num_degrees, num_channels),
            fiber_out=Fiber.create(1, num_out_channels),
            fiber_edge=Fiber.create(1, self.num_channels - 1), # So there are num_channels when dist is cat'ed
            **kwargs
        )
        self.mlp = nn.Sequential(
            nn.Linear(num_out_channels, num_out_channels),
            nn.SiLU(),
            nn.Linear(num_out_channels, 1)
        )
        self.pool_edges = SumPoolingEdges()

    def forward(self, graph, forces=True, create_graph=False, standardized=False):
        edge_energies = self.forward_edge_energies(graph)
        energies = self.pool_edges(graph, edge_energies)

        if not standardized: # Remove standardization
            energies = energies * self.energy_std
        if not forces:
            return energies
        forces = -torch.autograd.grad(torch.sum(energies),
                                      graph.ndata['pos'],
                                      create_graph=create_graph,
                                      )[0]
        return energies, forces

    def forward_edge_energies(self, graph):
        dist = torch.norm(graph.edata['rel_pos'], p=2, dim=1)
        scale = self.scale_fn(dist, self.cutoff)
        subscale = self.subscale_fn(dist, self.screen)

        species_embedding = self.embedding(graph.ndata['species'] - 1)  # -1 so H starts at 0
        node_feats = {'0': species_embedding.unsqueeze(-1)}
        radial_basis = self.get_radial_basis(dist)
        edge_feats = {'0': radial_basis.unsqueeze(-1)}

        feats = self.transformer(graph, node_feats, edge_feats, scale)
        learned_energies = self.mlp(feats).squeeze(-1) * scale
        coulomb_energies = self.screened_coulomb(graph, dist, subscale)
        return learned_energies + coulomb_energies

    @staticmethod
    def scale_fn(dist, cutoff):
        scale = torch.zeros_like(dist)
        bump_fn = lambda x : torch.exp(1 - 1 / (1 - x**2))
        scale[dist < cutoff] = bump_fn(2 * dist[dist < cutoff] / cutoff -1)
        return scale

    @staticmethod
    def subscale_fn(dist, cutoff):
        subscale = torch.zeros_like(dist)
        bump_fn = lambda x : torch.exp(1 - 1 / (1 - x**2))
        subscale[dist < cutoff] = bump_fn(dist[dist < cutoff] / cutoff)
        return subscale


    def screened_coulomb(self, graph, dist, scale):
        screened_energies = torch.zeros_like(dist)
        support = (scale > 0)
        Z = graph.ndata['species']
        u, v = graph.edges()
        energy_unit = self.coulomb_energy_unit / self.energy_std
        coulomb_energies = energy_unit * Z[u[support]] * Z[v[support]] / (2 * dist[support])  # / 2 for both directions
        screened_energies[support] = coulomb_energies * scale[support]  # Screen coulomb energies
        return screened_energies

    def get_radial_basis(self, dist) -> Tensor:
        gaussian_centers = torch.linspace(0, self.cutoff, self.num_channels-1, device=dist.device)
        dx = gaussian_centers[1] - gaussian_centers[0]
        diffs = dist[:,None] - gaussian_centers[None,:]
        return torch.exp(-2 * diffs**2 / dx**2)

    @staticmethod
    def loss_fn(pred, target):
        energy_loss = F.mse_loss(pred[0], target[0])
        forces_loss = F.mse_loss(pred[1], target[1])
        return energy_loss, forces_loss

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Model architecture")
        SE3TransformerTrIP.add_argparse_args(parser)
        parser.add_argument('--num_degrees',
                            help='Number of degrees to use. Hidden features will have types [0, ..., num_degrees - 1]',
                            type=int, default=3)
        parser.add_argument('--num_channels', help='Number of channels for the hidden features', type=int, default=16)
        return parent_parser
