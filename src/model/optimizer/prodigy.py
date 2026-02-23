from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


class Prodigy(Optimizer):
    def __init__(
        self,
        params,
        lr=1.0,
        betas=(0.9, 0.999),
        d_coef=0.8,
        weight_decay=0.01,
        eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            d_coef=d_coef,
            weight_decay=weight_decay,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            if "d" not in group:
                group["d"] = 1e-6
                group["d0"] = 1e-6
                group["s_k"] = 0.0

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["p0"] = p.clone()

                state["step"] += 1
                step = state["step"]

                if group["weight_decay"] > 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                update = exp_avg / denom

                group["s_k"] += torch.dot(update.flatten(), (p - state["p0"]).flatten()).item()
                if group["s_k"] > 0:
                    group["d"] = max(group["d"], group["d0"] + group["d_coef"] * group["s_k"])

                step_size = group["lr"] * group["d"] / bias_correction1
                p.add_(update, alpha=-step_size)

        return loss
