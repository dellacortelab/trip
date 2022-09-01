import torch as th
from torch import nn

from dgl import function as fn


# Compare to (old) dgl implementation of edge_softmax:
# https://docs.dgl.ai/en/0.2.x/_modules/dgl/nn/pytorch/softmax.html

class WeightedEdgeSoftmax(nn.Module):
    def __init__(self):
        super(WeightedEdgeSoftmax, self).__init__()
        # compute the softmax
        self._logits_name = "_logits"
        self._max_logits_name = "_max_logits"
        self._normalizer_name = "_norm"
        self._scale_name = "_scale"

    def forward(self, graph, logits, scale=None):
        graph.edata[self._logits_name] = logits

        # compute the softmax
        graph.update_all(fn.copy_edge(self._logits_name, self._logits_name),
                         fn.max(self._logits_name, self._max_logits_name))
        # minus the max and exp
        if scale is None:  # Use original code
            graph.apply_edges(
                lambda edges: {self._logits_name : th.exp(edges.data[self._logits_name] -
                                                        edges.dst[self._max_logits_name])})
        else:  # scale the exp
            graph.edata[self._scale_name] = scale.unsqueeze(-1)
            graph.apply_edges(
                lambda edges: {self._logits_name :
                    edges.data[self._scale_name] * th.exp(edges.data[self._logits_name] -
                                                        edges.dst[self._max_logits_name])})
        # compute normalizer
        graph.update_all(fn.copy_edge(self._logits_name, self._logits_name),
                         fn.sum(self._logits_name, self._normalizer_name))
        return graph.edata.pop(self._logits_name)

    def __repr__(self):
        return 'WeightedEdgeSoftmax()'
        