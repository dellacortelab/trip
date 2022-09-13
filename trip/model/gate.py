

from typing import Dict

import torch
from torch import nn
from torch import Tensor
from torch.cuda.nvtx import range as nvtx_range

from se3_transformer.model import Fiber


class TrIPGate(nn.Module):
    def __init__(self, fiber: Fiber, nonlinearity: nn.Module = nn.SiLU()):
        super(TrIPGate, self).__init__()
        self.fiber = fiber
        self.nonlinearity = nonlinearity
        channels = torch.tensor(fiber.channels)
        channels[0] -= torch.sum(channels[1:])  # Don't count gate scalars
        self.cumsum = [0] + torch.cumsum(channels, dim=0).tolist()

    def forward(self, features: Dict[str, Tensor], *args, **kwargs) -> Dict[str, Tensor]:
        with nvtx_range('TrIPGate'):
            output = {}
            for i, (degree, feat) in enumerate(features.items()):
                start = self.cumsum[i]
                stop = self.cumsum[i + 1]
                if i == 0:
                    scalars = self.nonlinearity(feat)
                    output[degree] = scalars[:, start:stop]
                else:
                    output[degree] = feat * scalars[:, start:stop]
        return output

    @staticmethod
    def make_input_fiber(fiber: Fiber):
        structure = {}
        for k, v in fiber.items():
            if k == 0 :
                structure[k] = sum(fiber.channels)
            else:
                structure[k] = v
        fiber_gate = Fiber(structure)
        return fiber_gate
