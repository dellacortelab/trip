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


class Sequential(nn.Sequential):
    """ Sequential module with arbitrary forward args and kwargs. Used to pass graph, basis and edge features. """

    def forward(self, input, *args, **kwargs):
        for module in self:
            input = module(input, *args, **kwargs)
        return input


def get_populated_edge_features(relative_pos: Tensor, edge_features: Optional[Dict[str, Tensor]] = None):
    """ Add relative positions to existing edge features """
    edge_features = edge_features.copy() if edge_features else {}
    r = relative_pos.norm(dim=-1, keepdim=True)
    if '0' in edge_features:
        edge_features['0'] = torch.cat([edge_features['0'], r[..., None]], dim=1)
    else:
        edge_features['0'] = r[..., None]

    return edge_features


class SE3Transformer(nn.Module):
    def __init__(self,
                 num_layers: int,
                 fiber_in: Fiber,
                 fiber_hidden: Fiber,
                 fiber_out: Fiber,
                 num_heads: int,
                 channels_div: int,
                 fiber_edge: Fiber = Fiber({}),
                 return_type: Optional[int] = None,
                 pooling: Optional[Literal['avg', 'max', 'sum']] = None,
                 norm: bool = True,
                 use_layer_norm: bool = True,
                 tensor_cores: bool = False,
                 low_memory: bool = False,
                 cutoff: float = float('inf'),
                 **kwargs):
        """
        :param num_layers:          Number of attention layers
        :param fiber_in:            Input fiber description
        :param fiber_hidden:        Hidden fiber description
        :param fiber_out:           Output fiber description
        :param fiber_edge:          Input edge fiber description
        :param num_heads:           Number of attention heads
        :param channels_div:        Channels division before feeding to attention layer
        :param return_type:         Return only features of this type
        :param pooling:             'avg', 'max', 'sum' graph pooling before MLP layers
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
        self.return_type = return_type
        self.pooling = pooling
        self.max_degree = max(*fiber_in.degrees, *fiber_hidden.degrees, *fiber_out.degrees)
        self.tensor_cores = tensor_cores
        self.low_memory = low_memory
        self.cutoff = cutoff

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
                                                   cutoff=cutoff))
            if norm:
                graph_modules.append(NormSE3(fiber_hidden))
            fiber_in = fiber_hidden

        graph_modules.append(ConvSE3(fiber_in=fiber_in,
                                     fiber_out=fiber_out,
                                     fiber_edge=fiber_edge,
                                     self_interaction=True,
                                     use_layer_norm=use_layer_norm,
                                     max_degree=self.max_degree))
        self.graph_modules = Sequential(*graph_modules)

        if pooling is not None:
            assert return_type is not None, 'return_type must be specified when pooling'
            self.pooling_module = GPooling(pool=pooling, feat_type=return_type)

    def forward(self, graph: DGLGraph, node_feats: Dict[str, Tensor],
                edge_feats: Optional[Dict[str, Tensor]] = None,
                basis: Optional[Dict[str, Tensor]] = None):
        # Compute bases in case they weren't precomputed as part of the data loading
        basis = basis or get_basis(graph.edata['rel_pos'], max_degree=self.max_degree, compute_gradients=False,
                                   use_pad_trick=self.tensor_cores and not self.low_memory,
                                   amp=torch.is_autocast_enabled())

        # Add fused bases (per output degree, per input degree, and fully fused) to the dict
        basis = update_basis_with_fused(basis, self.max_degree, use_pad_trick=self.tensor_cores and not self.low_memory,
                                        fully_fused=self.fuse_level == ConvSE3FuseLevel.FULL)

        edge_feats = get_populated_edge_features(graph.edata['rel_pos'], edge_feats)

        node_feats = self.graph_modules(node_feats, edge_feats, graph=graph, basis=basis)

        if self.pooling is not None:
            return self.pooling_module(node_feats, graph=graph)

        if self.return_type is not None:
            return node_feats[str(self.return_type)]

        return node_feats

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--num_layers', type=int, default=7,
                            help='Number of stacked Transformer layers')
        parser.add_argument('--num_heads', type=int, default=8,
                            help='Number of heads in self-attention')
        parser.add_argument('--channels_div', type=int, default=2,
                            help='Channels division before feeding to attention layer')
        parser.add_argument('--pooling', type=str, default='sum', const=None, nargs='?', choices=['max', 'avg', 'sum'],
                            help='Type of graph pooling')
        parser.add_argument('--norm', type=str2bool, nargs='?', const=True, default=False,
                            help='Apply a normalization layer after each attention block')
        parser.add_argument('--use_layer_norm', type=str2bool, nargs='?', const=True, default=False,
                            help='Apply layer normalization between MLP layers')
        parser.add_argument('--low_memory', type=str2bool, nargs='?', const=True, default=False,
                            help='If true, will use fused ops that are slower but that use less memory '
                                 '(expect 25 percent less memory). '
                                 'Only has an effect if AMP is enabled on Volta GPUs, or if running on Ampere GPUs')
        return parser


class SE3TransformerPooled(nn.Module):
    def __init__(self,
                 fiber_in: Fiber,
                 fiber_out: Fiber,
                 fiber_edge: Fiber,
                 num_degrees: int,
                 num_channels: int,
                 output_dim: int,
                 **kwargs):
        super().__init__()
        kwargs['pooling'] = kwargs['pooling'] or 'max'
        self.transformer = SE3Transformer(
            fiber_in=fiber_in,
            fiber_hidden=Fiber.create(num_degrees, num_channels),
            fiber_out=fiber_out,
            fiber_edge=fiber_edge,
            return_type=0,
            **kwargs
        )

        n_out_features = fiber_out.num_features
        self.mlp = nn.Sequential(
            nn.Linear(n_out_features, n_out_features),
            nn.ReLU(),
            nn.Linear(n_out_features, output_dim)
        )

    def forward(self, graph, node_feats, edge_feats, basis=None):
        feats = self.transformer(graph, node_feats, edge_feats, basis).squeeze(-1)
        y = self.mlp(feats).squeeze(-1)
        return y

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Model architecture")
        SE3Transformer.add_argparse_args(parser)
        parser.add_argument('--num_degrees',
                            help='Number of degrees to use. Hidden features will have types [0, ..., num_degrees - 1]',
                            type=int, default=4)
        parser.add_argument('--num_channels', help='Number of channels for the hidden features', type=int, default=32)
        return parent_parser


class TrIP(nn.Module):
    def __init__(self,
                 fiber_in: Fiber,
                 fiber_out: Fiber,
                 fiber_edge: Fiber,
                 num_degrees: int,
                 num_channels: int,
                 output_dim: int,
                 cutoff: float,
                 num_basis_fns: int,
                 **kwargs):
        super().__init__()
        kwargs['pooling'] = kwargs['pooling'] or 'max'
        self.transformer = SE3Transformer(
            fiber_in=fiber_in,
            fiber_hidden=Fiber.create(num_degrees, num_channels),
            fiber_out=fiber_out,
            fiber_edge=fiber_edge,
            return_type=0,
            **kwargs
        )
        self.cutoff = cutoff
        self.num_basis_fns = num_basis_fns

        n_out_features = fiber_out.num_features
        self.mlp = nn.Sequential(
            nn.Linear(n_out_features, n_out_features),
            nn.ReLU(),
            nn.Linear(n_out_features, output_dim)
        )

    def forward(self, inputs, forces=True, create_graph=True):
        graph, node_feats, *basis = inputs
        if forces==True:
            graph.ndata['pos'].requires_grad = True
            graph.edata['rel_pos'] = self._get_relative_pos(graph) # Calculate here so gradients go through the pos.
            tr = self.transformer
            basis = get_basis(graph.edata['rel_pos'], max_degree=tr.max_degree, compute_gradients=True,
                                       use_pad_trick=tr.tensor_cores and not tr.low_memory,
                                       amp=torch.is_autocast_enabled()
            )
        radial_basis = self._get_radial_basis(graph, self.cutoff, self.num_basis_fns)
        edge_feats = {'0': radial_basis[:,:,None]}
        feats = self.transformer(graph, node_feats, edge_feats, basis).squeeze(-1)
        energies = self.mlp(feats).squeeze(-1) 
        if not forces:
            return energies
        forces = -torch.autograd.grad(torch.sum(energies),
                                      graph.ndata['pos'],
                                      create_graph=create_graph,
                                     )[0]
        return energies, forces

    @staticmethod
    def _get_relative_pos(graph: DGLGraph) -> Tensor:
        x = graph.ndata['pos']
        src, dst = graph.edges()
        rel_pos = x[dst] - x[src]
        return rel_pos

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
        SE3TransformerPooled.add_argparse_args(parser)
        parser.add_argument('--num_basis_fns', help='Number of radial basis functions', type=int, default=16)
        return parent_parser
