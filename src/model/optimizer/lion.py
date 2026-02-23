from __future__ import annotations

import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                if wd > 0:
                    p.mul_(1 - lr * wd)

                c = exp_avg.clone().mul_(beta1).add_(grad, alpha=1 - beta1)
                p.add_(torch.sign(c), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
