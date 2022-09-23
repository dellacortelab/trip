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

from copy import deepcopy
import pathlib
from typing import Optional, Dict

from apex.optimizers import FusedAdam, FusedLAMB
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

import dgl
from dgl import DGLGraph

from se3_transformer.model.basis import get_basis, update_basis_with_fused
from se3_transformer.model.layers.attention import AttentionBlockSE3
from se3_transformer.model.layers.convolution import ConvSE3, ConvSE3FuseLevel
from se3_transformer.model.fiber import Fiber
from se3_transformer.model.transformer import get_populated_edge_features
from se3_transformer.runtime.utils import str2bool, get_local_rank

from trip.model.layers import TrIPNorm
from trip.model.layers import SumPoolingEdges
from trip.model.layers import WeightedEdgeSoftmax

        
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

        graph_modules = []
        for _ in range(num_layers):
            # TODO: Make fiber_conv and fiber_hidden distinction understandable
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
                graph_modules.append(TrIPNorm(fiber_hidden, nonlinearity = lambda x : x))
            fiber_in = fiber_hidden

        graph_modules.append(ConvSE3(fiber_in=fiber_in,
                                     fiber_out=fiber_out,
                                     fiber_edge=fiber_edge,
                                     self_interaction=False,
                                     pool=False,
                                     use_layer_norm=use_layer_norm,
                                     max_degree=self.max_degree))
        self.graph_modules = nn.ModuleList(graph_modules)

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

        feats = self.forward_graph_modules(node_feats, edge_feats, graph=graph, basis=basis, scale=scale)
        return feats['0'].squeeze(-1)

    def forward_graph_modules(self, node_feats, edge_feats, graph, basis, scale):
        for module in self.graph_modules:
            node_feats = module(node_feats, edge_feats, graph, basis, scale)
            if isinstance(module, AttentionBlockSE3):
                # Residual edge connection, allows more information to pass to radial profile
                num_edge_channels = min(node_feats['0'].shape[1], edge_feats['0'].shape[1]) 
                edge_feats['0'][:,:num_edge_channels] = edge_feats['0'][:,:num_edge_channels] + dgl.ops.copy_u(graph, node_feats['0'][:,:num_edge_channels])

        return node_feats

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


class TrIPModel(nn.Module):
    def __init__(self,
                 num_degrees: int,
                 num_channels: int,
                 energy_std: float,
                 screen: float,
                 cutoff: float,
                 coulomb_energy_unit: float,
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
        coulomb_factor = Z[u[support]] * Z[v[support]] / (2 * dist[support])  # / 2 for both directions
        screened_energies[support] = energy_unit * coulomb_factor * scale[support]  # Screen coulomb energies
        return screened_energies

    def get_radial_basis(self, dist) -> Tensor:
        gaussian_centers = torch.linspace(0, self.cutoff,
                                          self.num_channels-1, device=dist.device)
        dx = gaussian_centers[1] - gaussian_centers[0]
        diffs = dist[:,None] - gaussian_centers[None,:]
        return torch.exp(-2 * diffs**2 / dx**2)

    @staticmethod
    def add_argparse_args(parser):
        SE3TransformerTrIP.add_argparse_args(parser)
        parser.add_argument('--coulomb_energy_unit',
                            help='Value of e^2/(4*pi*e0) in preferred units, default is Ha*A',
                            type=float, default=0.529_177_211)  # e^2/(4*pi*e0) in Ha*A)
        parser.add_argument('--num_degrees',
                            help='Number of degrees to use. Hidden features will have types [0, ..., num_degrees - 1]',
                            type=int, default=3)
        parser.add_argument('--num_channels',
                            help='Number of channels for the hidden features',
                            type=int, default=16)
        return parser


class TrIP(nn.Module):
    def __init__(self,
                **kwargs):
        super().__init__()
        self.model = TrIPModel(**kwargs)
        self.kwargs = deepcopy(kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def save(self, optimizer: Optimizer, epoch: int, path: pathlib.Path):
        """ Saves model, optimizer and epoch states to path (only once per node) """
        if get_local_rank() == 0:
            checkpoint = {
                'state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'kwargs': self.kwargs,
                'epoch': epoch
            }
            torch.save(checkpoint, str(path))
        return checkpoint

    @staticmethod
    def load(path: pathlib.Path, map_location='cuda:0', optimizer=False):
        """ Loads model, optimizer and epoch states from path """
        checkpoint = torch.load(str(path), map_location=map_location)
        kwargs = checkpoint['kwargs']
        model = TrIP(**kwargs)

        model.load_state_dict(checkpoint['state_dict'])

        if not optimizer:
            return model
        
        optimizer = TrIP.make_optimizer(model, **kwargs)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer

    def load_state(self, path: pathlib.Path, map_location, optimizer=False, checkpoint=None):
        if checkpoint is None:
            checkpoint = torch.load(str(path), map_location=map_location)
        self.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']

    @staticmethod
    def make_optimizer(model, optimizer, learning_rate, momentum, weight_decay, **kwargs):
        parameters = TrIP.add_weight_decay(model, weight_decay)
        if optimizer == 'adam':
            optimizer = FusedAdam(parameters, lr=learning_rate, betas=(momentum, 0.999),
                                weight_decay=weight_decay)
        elif optimizer == 'lamb':
            optimizer = FusedLAMB(parameters, lr=learning_rate, betas=(momentum, 0.999),
                                weight_decay=weight_decay)
        else:
            optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum,
                                        weight_decay=weight_decay)
        return optimizer

    @staticmethod
    # https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3
    def add_weight_decay(model, weight_decay, skip_list=[]):
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]

    @staticmethod
    def loss_fn(pred, target):
        energy_loss = F.mse_loss(pred[0], target[0])
        forces_loss = F.mse_loss(pred[1], target[1])
        return energy_loss, forces_loss

    def add_argparse_args(parent_parser):
        optimizer = parent_parser.add_argument_group('Optimizer')
        optimizer.add_argument('--optimizer', choices=['adam', 'sgd', 'lamb'], default='adam')
        optimizer.add_argument('--learning_rate', '--lr', dest='learning_rate', type=float, default=0.002)
        optimizer.add_argument('--gamma', type=float, default=1.0)
        optimizer.add_argument('--momentum', type=float, default=0.9)
        optimizer.add_argument('--weight_decay', type=float, default=0.1)

        parser = parent_parser.add_argument_group("Model architecture")
        parser.add_argument('--force_weight', type=float, default=0.1,
                            help='Weigh force losses to energy losses')
        parser.add_argument('--cutoff', type=float, default=4.6,
                            help='Radius of graph neighborhood')
        parser.add_argument('--screen', type=float, default=1.0,
                            help='Distance where Coloumb force between nuclei begins dominating')
        TrIPModel.add_argparse_args(parser)
        return parent_parser
