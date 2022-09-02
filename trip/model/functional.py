from typing import List, Optional

import torch
from torch import Tensor
from torch.nn.functional import _verify_batch_size


from torch.overrides import has_torch_function_variadic

def layer_norm(
    input: Tensor,
    normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    r"""Applies Layer Normalization for last certain number of dimensions.

    See :class:`~torch.nn.LayerNorm` for details.
    """
    if has_torch_function_variadic(input, weight, bias):
        return handle_torch_function(
            layer_norm, (input, weight, bias), input, normalized_shape, weight=weight, bias=bias, eps=eps
        )
    return torch.layer_norm(input, normalized_shape, weight, bias, eps, torch.backends.cudnn.enabled)


def group_norm(
    input: Tensor, num_groups: int, weight: Optional[Tensor] = None, bias: Optional[Tensor] = None, eps: float = 1e-5
) -> Tensor:
    r"""Applies Group Normalization for last certain number of dimensions.

    See :class:`~torch.nn.GroupNorm` for details.
    """
    if has_torch_function_variadic(input, weight, bias):
        return handle_torch_function(group_norm, (input, weight, bias,), input, num_groups, weight=weight, bias=bias, eps=eps)
    _verify_batch_size([input.size(0) * input.size(1) // num_groups, num_groups] + list(input.size()[2:]))
    return torch.group_norm(input, num_groups, weight, bias, eps, torch.backends.cudnn.enabled)