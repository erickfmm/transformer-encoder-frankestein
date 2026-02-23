from __future__ import annotations

import torch
from torch.optim import Optimizer


class LAMB(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
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
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_c1 = 1 - beta1 ** state["step"]
                bias_c2 = 1 - beta2 ** state["step"]
                update = (exp_avg / bias_c1) / ((exp_avg_sq / bias_c2).sqrt() + eps)
                if weight_decay != 0:
                    update = update + weight_decay * p

                w_norm = torch.norm(p).clamp(min=1e-12)
                u_norm = torch.norm(update).clamp(min=1e-12)
                trust_ratio = (w_norm / u_norm).item()
                p.add_(update, alpha=-lr * trust_ratio)

        return loss
