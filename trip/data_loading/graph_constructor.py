


from typing import List

import torch
from torch import Tensor

import dgl

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
                graphs.append(self._create_box_graph(pos, self.cutoff))

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
        adj_dir = GraphConstructor._get_adj_dir()
        pos_copy = pos[None, :, :] + adj_dir[:, None, :] * box_size[None, :]
        dist_mat = torch.cdist(pos[None, ...], pos_copy,
                               compute_mode='donot_use_mm_for_euclid_dist')
        adj_mat = dist_mat < cutoff
        adj_mat[0].fill_diagonal_(0)
        _, u, v = torch.nonzero(adj_mat, as_tuple=True)
        graph = dgl.graph((u, v), num_nodes=len(pos))
        graph.ndata['pos'] = pos
        return graph

    @staticmethod
    def _get_adj_dir():
        arange = torch.arange(27) + 13
        rel_pos = torch.stack([arange//9, arange//3, arange//1]).T
        rel_pos = (rel_pos % 3) - 1
        return rel_pos

    @staticmethod
    def _set_rel_pos(batched_graph, box_size_list):
        # Gradients need to evaluate back to pos for forces
        batched_graph.ndata['pos'].requires_grad_(True)
        src, dst = batched_graph.edges()
        pos = batched_graph.ndata['pos']
        rel_pos = pos[dst] - pos[src]

        # Fix the rel_pos for boxes using PBC
        num_nodes = batched_graph.batch_num_nodes()
        cum_num_nodes = torch.cumsum(num_nodes, dim=0)
        for i, box_size in enumerate(box_size_list):
            if not torch.isinf(box_size).all():  # Check if subgraph is periodic
                stop = cum_num_nodes[i]
                start = stop - num_nodes[i]
                # Periodic boundary condition
                shifted_rel_pos = rel_pos[start:stop] + box_size/2
                rel_pos[start:stop] = shifted_rel_pos%box_size - box_size/2

        batched_graph.edata['rel_pos'] = rel_pos
        return batched_graph
