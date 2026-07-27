import torch
from torch.optim.optimizer import Optimizer, required


class FedProxOptimizer(Optimizer):
    """FedAvg/FedProx local solver retained for ClientFedAvg compatibility."""

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        variance=0,
        mu=0,
    ):
        self.itr = 0
        self.a_sum = 0
        self.mu = mu
        if lr is not required and lr < 0.0:
            raise ValueError(f"invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum and zero dampening")
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "variance": variance,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.data
                if group["weight_decay"] != 0:
                    gradient.add_(parameter.data, alpha=group["weight_decay"])
                state = self.state[parameter]
                if "old_init" not in state:
                    state["old_init"] = torch.clone(parameter.data).detach()
                if group["momentum"] != 0:
                    if "momentum_buffer" not in state:
                        buffer = state["momentum_buffer"] = torch.clone(
                            gradient
                        ).detach()
                    else:
                        buffer = state["momentum_buffer"]
                        buffer.mul_(group["momentum"]).add_(
                            gradient, alpha=1 - group["dampening"]
                        )
                    if group["nesterov"]:
                        gradient = gradient.add(
                            buffer, alpha=group["momentum"]
                        )
                    else:
                        gradient = buffer
                gradient.add_(
                    parameter.data - state["old_init"], alpha=self.mu
                )
                parameter.data.add_(gradient, alpha=-group["lr"])
        return loss


__all__ = ["FedProxOptimizer"]
