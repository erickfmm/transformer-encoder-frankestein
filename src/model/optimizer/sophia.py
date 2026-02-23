from __future__ import annotations

import torch
from torch.optim import Optimizer


class Sophia(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.965, 0.99),
        rho=0.04,
        weight_decay=1e-1,
        update_k=10,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            rho=rho,
            weight_decay=weight_decay,
            update_k=update_k,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, hessian_estimate=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["hessian"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                hessian = state["hessian"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                if hessian_estimate is not None and step % int(group["update_k"]) == 1:
                    est = hessian_estimate.get(p) if isinstance(hessian_estimate, dict) else None
                    if est is not None:
                        hessian.mul_(beta2).add_(est, alpha=1 - beta2)

                p.mul_(1 - group["lr"] * group["weight_decay"])

                h_max = torch.max(group["rho"] * hessian, torch.full_like(p, 1e-15))
                update = exp_avg / h_max
                update.clamp_(-1.0, 1.0)
                p.add_(update, alpha=-group["lr"])

        return loss
