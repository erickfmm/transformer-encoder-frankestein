from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


class GaLoreAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        rank=128,
        update_proj_gap=200,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
    ):
        defaults = dict(
            lr=lr,
            rank=rank,
            update_proj_gap=update_proj_gap,
            betas=betas,
            eps=eps,
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
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0

                state["step"] += 1
                step = state["step"]

                if group["weight_decay"] > 0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])

                g_low = grad
                if grad.dim() == 2:
                    if "projector" not in state or step % int(group["update_proj_gap"]) == 1:
                        u, _, vh = torch.linalg.svd(grad.float(), full_matrices=False)
                        rank = min(int(group["rank"]), min(grad.shape))
                        if grad.shape[0] >= grad.shape[1]:
                            state["projector"] = u[:, :rank].to(grad.dtype)
                            state["proj_type"] = "left"
                        else:
                            state["projector"] = vh[:rank, :].to(grad.dtype)
                            state["proj_type"] = "right"

                    pmat = state["projector"]
                    if state["proj_type"] == "left":
                        g_low = pmat.T @ grad
                    else:
                        g_low = grad @ pmat.T

                if "exp_avg" not in state or state["exp_avg"].shape != g_low.shape:
                    state["exp_avg"] = torch.zeros_like(g_low)
                    state["exp_avg_sq"] = torch.zeros_like(g_low)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(g_low, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_low, g_low, value=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = group["lr"] / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                update_low = exp_avg / denom

                if grad.dim() == 2:
                    pmat = state["projector"]
                    if state["proj_type"] == "left":
                        update = pmat @ update_low
                    else:
                        update = update_low @ pmat
                else:
                    update = update_low

                p.add_(update, alpha=-step_size)

        return loss
