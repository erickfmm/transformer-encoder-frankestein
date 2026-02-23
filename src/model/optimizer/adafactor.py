from __future__ import annotations

import torch
from torch.optim import Optimizer


class Adafactor(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        beta2_decay=0.8,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
    ):
        defaults = dict(
            lr=lr,
            beta2_decay=beta2_decay,
            eps=eps,
            clip_threshold=clip_threshold,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            eps1, eps2 = group["eps"]
            clip_threshold = float(group["clip_threshold"])

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    if grad.dim() >= 2:
                        state["exp_avg_sq_row"] = torch.zeros(
                            grad.shape[:-1], dtype=grad.dtype, device=grad.device
                        )
                        state["exp_avg_sq_col"] = torch.zeros(
                            grad.shape[:-2] + (grad.shape[-1],), dtype=grad.dtype, device=grad.device
                        )
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                state["step"] += 1
                t = state["step"]
                beta2 = 1.0 - (t ** -float(group["beta2_decay"]))

                grad_sq = grad.pow(2).add(eps1)

                if grad.dim() >= 2:
                    row_sums = torch.mean(grad_sq, dim=-1)
                    col_sums = torch.mean(grad_sq, dim=-2)

                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2).add_(row_sums, alpha=1.0 - beta2)
                    exp_avg_sq_col.mul_(beta2).add_(col_sums, alpha=1.0 - beta2)

                    row_avg = exp_avg_sq_row / (1.0 - beta2 ** t)
                    col_avg = exp_avg_sq_col / (1.0 - beta2 ** t)

                    row_avg_mean = torch.mean(row_avg, dim=-1, keepdim=True)
                    v = (row_avg.unsqueeze(-1) * col_avg.unsqueeze(-2)) / (row_avg_mean + eps1)
                    update = grad / torch.sqrt(v + eps2)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).add_(grad_sq, alpha=1.0 - beta2)
                    v = exp_avg_sq / (1.0 - beta2 ** t)
                    update = grad / torch.sqrt(v + eps2)

                rms = torch.norm(update) / (update.numel() ** 0.5)
                divisor = max(1.0, float(rms) / clip_threshold)
                update.div_(divisor)

                p.add_(update, alpha=-lr)

        return loss
