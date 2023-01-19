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
from typing import Dict

from apex.optimizers import FusedAdam, FusedLAMB
import torch
from torch import Tensor
import torch.nn as nn

import dgl
from dgl import DGLGraph
from dgl.nn import SumPooling

from se3_transformer.model.basis import get_basis, update_basis_with_fused
from se3_transformer.model.layers.attention import AttentionBlockSE3
from se3_transformer.model.layers.convolution import ConvSE3, ConvSE3FuseLevel
from se3_transformer.model.fiber import Fiber
from se3_transformer.model.transformer import get_populated_edge_features
from se3_transformer.runtime.utils import str2bool

from trip.model.layers import TrIPNorm
from trip.model.layers import WeightedEdgeSoftmax

        
class TrIPTransformer(nn.Module):
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
        self.low_memory = low_memory  # TODO: Check whether low_memory works now or should remove

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
                                     self_interaction=True,
                                     pool=True,
                                     use_layer_norm=use_layer_norm,
                                     max_degree=self.max_degree))
        self.graph_modules = nn.ModuleList(graph_modules)

    def forward(self,
                graph: DGLGraph,
                node_feats: Dict[str, Tensor],
                edge_feats: Dict[str, Tensor],
                scale: Tensor):
        # Compute bases in case they weren't precomputed as part of the data loading
        basis = get_basis(graph.edata['rel_pos'], max_degree=self.max_degree, compute_gradients=True,
                                   use_pad_trick=self.tensor_cores and not self.low_memory,
                                   amp=torch.is_autocast_enabled())
        
        # Add fused bases (per output degree, per input degree, and fully fused) to the dict
        basis = update_basis_with_fused(basis, self.max_degree, use_pad_trick=self.tensor_cores and not self.low_memory,
                                        fully_fused=self.fuse_level == ConvSE3FuseLevel.FULL)

        # Scale basis with scale tensor
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
                edge_feats['0'] = edge_feats['0'][:,:num_edge_channels] + dgl.ops.copy_u(graph, node_feats['0'][:,:num_edge_channels])

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


class TrIPModel(TrIPTransformer):
    def __init__(self,
                 num_degrees: int,
                 num_channels: int,
                 energy_std: float,
                 cutoff: float,
                 coulomb_energy_unit: float,
                 **kwargs):
        self.num_degrees = num_degrees
        self.num_channels = num_channels
        self.energy_std = energy_std
        self.cutoff = cutoff
        self.coulomb_energy_unit = coulomb_energy_unit

        num_out_channels = num_channels * num_degrees
        super().__init__(   
            fiber_in=Fiber.create(1, num_channels),
            fiber_hidden=Fiber.create(num_degrees, num_channels),
            fiber_out=Fiber.create(1, num_out_channels),
            fiber_edge=Fiber.create(1, self.num_channels - 1), # So there are num_channels when dist is cat'ed
            **kwargs
            )
        self.embedding = nn.Embedding(100, num_channels)
        self.mlp = nn.Sequential(
            nn.Linear(num_out_channels + num_channels, num_out_channels),
            nn.SiLU(),
            nn.Linear(num_out_channels, num_out_channels),
            nn.SiLU(),
            nn.Linear(num_out_channels, 1)
        )
        self.pool = SumPooling()

    def forward(self, graph, forces=True, create_graph=False, standardized=False):
        atom_energies = self.forward_atom_energies(graph, standardized)
        energies = self.pool(graph, atom_energies)

        if not forces:
            return energies
        forces = -torch.autograd.grad(torch.sum(energies),
                                      graph.ndata['pos'],
                                      create_graph=create_graph,
                                      )[0]
        return energies, forces

    def forward_atom_energies(self, graph, standardized):
        dist = torch.norm(graph.edata['rel_pos'], p=2, dim=1)
        scale = self.scale_fn(dist, self.cutoff)

        species_embedding = self.embedding(graph.ndata['species'] - 1)  # -1 so H starts at 0
        node_feats = {'0': species_embedding.unsqueeze(-1)}
        radial_basis = self.get_radial_basis(dist)
        edge_feats = {'0': radial_basis.unsqueeze(-1)}

        feats = super().forward(graph, node_feats, edge_feats, scale)
        cat_feats = torch.cat([species_embedding, feats], dim=1)
        learned_energies = self.mlp(cat_feats).squeeze(-1)
        coulomb_energies = self.screened_coulomb(graph, dist, scale)
        atom_energies = learned_energies + coulomb_energies
        if standardized:
            return atom_energies

        # Return relative energies if requested
        atom_energies = atom_energies * self.energy_std
        if hasattr(self, 'si_tensor'): 
        # Add back si_energies if they exist. Used for testing but not training
            species = graph.ndata['species']
            si_energies = self.si_tensor[(species-1).tolist()]  # -1 so H starts at 0
            atom_energies = atom_energies + si_energies
        
        return atom_energies

    @staticmethod
    def scale_fn(dist, cutoff):
        scale = torch.zeros_like(dist)
        scale[dist < cutoff] = TrIPModel.bump_fn(dist[dist<cutoff] / cutoff, k=3)
        return scale

    @staticmethod
    def bump_fn(x, k=1):
        return torch.exp(k - k / (1 - x**2))

    def screened_coulomb(self, graph, dist, scale):
        Z = graph.ndata['species']
        u, v = graph.edges()
        Z_u, Z_v = Z[u], Z[v]
        raw_energies = self.coulomb_energy_unit * Z_u * Z_v / (2 * dist)  # / 2 for both directions
        screen = TrIPModel.zbl_screening_fn(dist, Z_u, Z_v)
        screened_energies = raw_energies * screen * scale  # Apply screening and cutoff function
        atom_coulomb_energies = dgl.ops.copy_e_sum(graph, screened_energies)  # Pool energies from edges to nodes
        return atom_coulomb_energies / self.energy_std

    @staticmethod
    def zbl_screening_fn(dist, Z_u, Z_v):
        # Universal screening function from "The Stopping and Range of Ions in Solids" by Ziegler, J.F., et al. (1985)
        au = 0.8854 * 0.529 / (Z_u**0.23 + Z_v**0.23)  # Universal screening length
        x = dist / au
        screening_factor = 0.1818*torch.exp(-3.2*x)\
                           + 0.5099*torch.exp(-0.9423*x)\
                           + 0.2802*torch.exp(-0.4028*x)\
                           + 0.02817*torch.exp(-0.2016*x)
        return screening_factor

    def get_radial_basis(self, dist) -> Tensor:
        return torch.zeros(len(dist), self.num_channels-1, device=dist.device, dtype=dist.dtype)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--num_degrees',
                            help='Number of degrees to use. Hidden features will have types [0, ..., num_degrees - 1]',
                            type=int, default=3)
        parser.add_argument('--num_channels',
                            help='Number of channels for the hidden features',
                            type=int, default=16)
        parser.add_argument('--cutoff',
                            help='Radius of graph neighborhood',
                            type=float, default=4.6)
        parser.add_argument('--coulomb_energy_unit',
                            help='Value of e^2/(4*pi*e0) in preferred units, default is Ha*A',
                            type=float, default=0.529)  # e^2/(4*pi*e0) in Ha*A)
        TrIPTransformer.add_argparse_args(parser)
        return parser


class TrIP(TrIPModel):
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        self.optimizer = TrIP.make_optimizer(self, **kwargs)
        self.kwargs = deepcopy(kwargs)

    def save(self, path: pathlib.Path, epoch: int):
        """ Saves model, optimizer and epoch states to path"""
        checkpoint = self.get_checkpoint(epoch)
        torch.save(checkpoint, str(path))

    def checkpoint(self, epoch: int):
        checkpoint = {
            'state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'kwargs': self.kwargs,
            'epoch': epoch
        }
        return checkpoint

    def load_state(self, checkpoint, map_location='cuda:0'):
        if isinstance(checkpoint, pathlib.Path) or isinstance(checkpoint, str):
            checkpoint = torch.load(str(checkpoint), map_location=map_location)
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.kwargs = deepcopy(checkpoint['kwargs'])
        return checkpoint

    @staticmethod
    def load(path: pathlib.Path, map_location='cuda:0'):
        """ Loads model, optimizer and epoch states from path """
        checkpoint = torch.load(str(path), map_location=map_location)
        kwargs = checkpoint['kwargs']
        model = TrIP(**kwargs)
        model.to(device=map_location)
        model.load_state(checkpoint, map_location)
        return model

    @staticmethod
    def make_optimizer(model, optimizer_type, learning_rate, momentum, weight_decay, **kwargs):
        parameters = TrIP.add_weight_decay(model, weight_decay, skip_list=['model.embedding.weight',
                                                                           'model.mlp.2.weight'])
        if optimizer_type == 'adam':
            return FusedAdam(parameters, lr=learning_rate, betas=(momentum, 0.999),
                             weight_decay=weight_decay)
        elif optimizer_type == 'lamb':
            return FusedLAMB(parameters, lr=learning_rate, betas=(momentum, 0.999),
                             weight_decay=weight_decay)
        else:
            return torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum,
                                   weight_decay=weight_decay)

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
    def loss_fn(pred, target, beta=1e-1):  # Similar to Huberloss, but smoother
        calc_loss = lambda x : beta * (torch.mean(torch.sqrt(x + beta**2/4)) - beta/2)  # Approximates mean(x) when x << beta, approximates beta*mean(sqrt(x)) when x >> beta
        energy_loss = calc_loss((pred[0] - target[0])**2)
        forces_loss = calc_loss(torch.sum((pred[1] - target[1])**2, dim=1))  # Argument is squared norm
        return energy_loss, forces_loss

    @staticmethod
    def error_fn(graph, pred, target):
        lens = torch.tensor([g.num_nodes() for g in dgl.unbatch(graph)])
        num_species = torch.sum(lens == 1)
        energy_error = (pred[0][:-num_species] - target[0][:-num_species])**2
        forces_error = torch.sum((pred[1][:-num_species] - target[1][:-num_species])**2, dim=1)
        return energy_error, forces_error

    @staticmethod
    def add_argparse_args(parent_parser):
        opt_parser = parent_parser.add_argument_group('Optimizer')
        opt_parser.add_argument('--optimizer_type', choices=['adam', 'sgd', 'lamb'], default='adam')
        opt_parser.add_argument('--learning_rate', '--lr', dest='learning_rate', type=float, default=0.002)
        opt_parser.add_argument('--gamma', type=float, default=1.0)
        opt_parser.add_argument('--momentum', type=float, default=0.9)
        opt_parser.add_argument('--weight_decay', type=float, default=0.1)

        model_parser = parent_parser.add_argument_group("Model architecture")
        model_parser.add_argument('--force_weight', type=float, default=0.1,
                                  help='Weigh force losses to energy losses')
        TrIPModel.add_argparse_args(model_parser)
        return parent_parser
