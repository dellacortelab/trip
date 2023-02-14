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

import numpy as np
from typing import List

import torch
from torch import Tensor

import dgl

from scipy.spatial import KDTree


class GraphConstructor:
    def __init__(self, cutoff: float):
        self.cutoff = cutoff

    def create_graphs(self,
                      pos_list: List[Tensor],
                      box_size_list : List[Tensor]):
        # Provide support when creating a single graph
        if not isinstance(pos_list, list):
            pos_list = [pos_list]
            box_size_list = [box_size_list]

        # Create graphs
        graphs = []
        for pos, box_size in zip(pos_list, box_size_list):
            if torch.isinf(box_size).all():
                graphs.append(self._create_vacuum_graph(pos, self.cutoff))
            else:
                graphs.append(self._create_box_graph(pos, box_size, self.cutoff))

        # Batched graphs and set relative positions
        batched_graph = dgl.batch(graphs)
        batched_graph = self._set_rel_pos(batched_graph, box_size_list)
        return batched_graph

    @staticmethod
    def _create_vacuum_graph(pos: Tensor, cutoff: float):
        dist_mat = torch.cdist(pos[None, ...], pos[None, ...],
                               compute_mode='donot_use_mm_for_euclid_dist'
                               ).squeeze(0)
        adj_mat = (dist_mat < cutoff).fill_diagonal_(False)
        u, v = torch.nonzero(adj_mat, as_tuple=True)
        graph = dgl.graph((u, v), num_nodes=len(pos))
        graph.ndata['pos'] = pos
        return graph

    @staticmethod
    def _create_box_graph(pos: Tensor, box_size: Tensor, cutoff: float):
        pos_np = pos.cpu().numpy().astype(np.double)
        box_size_np = box_size.cpu().numpy().astype(np.double)
        pos_np %= box_size_np
        tree = KDTree(pos_np, boxsize=box_size_np)
        pairs = tree.query_pairs(r=cutoff)
        u, v = torch.tensor(list(pairs)).T
        u, v = torch.cat((u,v)), torch.cat((v,u))  # Symmetrize graph
        graph = dgl.graph((u, v), num_nodes=len(pos), device=pos.device)
        graph.ndata['pos'] = pos
        return graph

    @staticmethod
    def _set_rel_pos(batched_graph, box_size_list):
        # Gradients need to evaluate back to pos for forces
        batched_graph.ndata['pos'].requires_grad_(True)
        src, dst = batched_graph.edges()
        pos = batched_graph.ndata['pos']
        rel_pos = pos[dst] - pos[src]

        # Fix the rel_pos for boxes using PBC
        num_edges = batched_graph.batch_num_edges()
        cum_num_edges = torch.cumsum(num_edges, dim=0)
        for i, box_size in enumerate(box_size_list):
            if not torch.isinf(box_size).all():  # Check if subgraph is periodic
                stop = cum_num_edges[i]
                start = stop - num_edges[i]
                # Periodic boundary condition
                shifted_rel_pos = rel_pos[start:stop] + box_size/2
                rel_pos[start:stop] = shifted_rel_pos%box_size - box_size/2

        batched_graph.edata['rel_pos'] = rel_pos
        return batched_graph
