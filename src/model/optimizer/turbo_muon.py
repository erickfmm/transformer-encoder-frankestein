from __future__ import annotations

import torch
from torch.optim import Optimizer


class TurboMuon(Optimizer):
    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=4,
        ns_eps=1e-7,
        weight_decay=0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_eps=ns_eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p.ndim < 2:
                    continue

                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(grad)
                g = grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf

                if group["weight_decay"] > 0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])

                a, b, c = 3.4445, -4.7750, 2.0315
                x = g.bfloat16()

                transposed = False
                if x.size(0) > x.size(1):
                    x = x.T
                    transposed = True

                a_mat = x @ x.T
                s = torch.sum(torch.abs(a_mat), dim=0, keepdim=True).clamp(min=group["ns_eps"]).pow(-0.5)
                x = x * s
                a_mat = a_mat * s * s.T

                for i in range(int(group["ns_steps"])):
                    b_mat = b * a_mat + c * (a_mat @ a_mat)
                    x = a * x + b_mat @ x
                    if i < int(group["ns_steps"]) - 1:
                        a_mat = x @ x.T

                if transposed:
                    x = x.T

                p.add_(x.to(p.dtype), alpha=-group["lr"])

        return loss
