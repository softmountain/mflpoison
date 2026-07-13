from typing import Iterable, List, Tuple

import torch


def all_finite(tensors: Iterable[torch.Tensor]) -> bool:
    return all(tensor is None or bool(torch.isfinite(tensor).all()) for tensor in tensors)


def nonfinite_gradient_names(named_parameters: Iterable[Tuple[str, torch.nn.Parameter]]) -> List[str]:
    return [
        name
        for name, parameter in named_parameters
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
    ]
