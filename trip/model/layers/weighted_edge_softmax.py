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

import torch as th
from torch import nn

from dgl import function as fn


class WeightedEdgeSoftmax(nn.Module):
    # Compare to (old) dgl implementation of edge_softmax:
    # https://docs.dgl.ai/en/0.2.x/_modules/dgl/nn/pytorch/softmax.html
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
        