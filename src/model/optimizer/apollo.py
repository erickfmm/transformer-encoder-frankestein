from __future__ import annotations

import math
import torch
from torch.optim import Optimizer

# Simple LCG constants for deterministic projector seed advancement per-parameter.
LCG_MULTIPLIER = 1103515245
LCG_INCREMENT = 12345
LCG_MODULUS = 2**31 - 1
# Fira-style norm-growth limiter threshold.
NORM_LIMITER_THRESHOLD = 1.01
EPS = 1e-8


class Apollo(Optimizer):
    """
    APOLLO optimizer: Adam-style low-rank auxiliary moments with projected
    gradient scaling back in the original parameter space.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        rank=128,
        update_proj_gap=200,
        scale=1.0,
        scale_type="channel",
        proj_type="std",
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        correct_bias=True,
        scale_front=False,
        disable_nl=False,
    ):
        if scale_type not in {"channel", "tensor"}:
            raise ValueError("scale_type must be one of: channel, tensor")
        if proj_type not in {"std", "reverse_std", "left", "right"}:
            raise ValueError("proj_type must be one of: std, reverse_std, left, right")

        defaults = dict(
            lr=lr,
            rank=rank,
            update_proj_gap=update_proj_gap,
            scale=scale,
            scale_type=scale_type,
            proj_type=proj_type,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
            scale_front=scale_front,
            disable_nl=disable_nl,
        )
        super().__init__(params, defaults)

    def _load_moments(self, state, grad_like):
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg is None or exp_avg_sq is None or exp_avg.shape != grad_like.shape:
            exp_avg = torch.zeros_like(grad_like)
            exp_avg_sq = torch.zeros_like(grad_like)
        return exp_avg, exp_avg_sq

    def _store_moments(self, state, exp_avg, exp_avg_sq):
        state["exp_avg"] = exp_avg
        state["exp_avg_sq"] = exp_avg_sq

    @staticmethod
    def _sample_projector(shape, rank, side, *, device, dtype, seed):
        generator = torch.Generator(device=device).manual_seed(int(seed))
        if side == "left":
            mat = torch.randn((shape[0], rank), generator=generator, device=device, dtype=dtype)
        else:
            mat = torch.randn((rank, shape[1]), generator=generator, device=device, dtype=dtype)
        return mat / math.sqrt(max(rank, 1))

    def _project_grad(self, grad, state, group):
        rank = max(1, min(int(group["rank"]), min(grad.shape)))
        step = int(state.get("step", 0))
        gap = max(1, int(group["update_proj_gap"]))
        proj_type = str(group["proj_type"])

        if proj_type == "std":
            side = "right" if grad.shape[0] >= grad.shape[1] else "left"
        elif proj_type == "reverse_std":
            side = "left" if grad.shape[0] >= grad.shape[1] else "right"
        elif proj_type in {"left", "right"}:
            side = proj_type
        else:
            side = "right"

        if "projector_seed" not in state:
            state["projector_seed"] = int(torch.randint(1, 2**31 - 1, (1,), device="cpu").item())

        needs_update = (
            "projector" not in state
            or state.get("projector_side") != side
            or step % gap == 0
            or state["projector"].shape[0] != (grad.shape[0] if side == "left" else rank)
            or state["projector"].shape[1] != (rank if side == "left" else grad.shape[1])
        )
        if needs_update:
            state["projector"] = self._sample_projector(
                grad.shape,
                rank,
                side,
                device=grad.device,
                dtype=grad.dtype,
                seed=state["projector_seed"],
            )
            state["projector_side"] = side
            state["projector_seed"] = int(
                (state["projector_seed"] * LCG_MULTIPLIER + LCG_INCREMENT) % LCG_MODULUS
            )

        proj = state["projector"]
        if side == "right":
            low_grad = grad @ proj.T
        else:
            low_grad = proj.T @ grad
        norm_dim = 0 if low_grad.shape[0] < low_grad.shape[1] else 1
        return low_grad, norm_dim

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
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Apollo does not support sparse gradients")

                state = self.state[p]
                state["step"] = int(state.get("step", 0)) + 1
                step_num = state["step"]

                projected = grad.ndim == 2 and int(group["rank"]) > 0
                if projected:
                    low_grad, norm_dim = self._project_grad(grad, state, group)
                else:
                    low_grad = grad
                    norm_dim = None

                exp_avg, exp_avg_sq = self._load_moments(state, low_grad)
                exp_avg.mul_(beta1).add_(low_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(low_grad, low_grad, value=1.0 - beta2)
                self._store_moments(state, exp_avg, exp_avg_sq)

                denom = exp_avg_sq.sqrt().add_(eps)
                norm_grad = exp_avg / denom

                step_size = lr
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1**step_num
                    bias_correction2 = 1.0 - beta2**step_num
                    step_size = lr * math.sqrt(bias_correction2) / max(bias_correction1, 1e-16)

                if projected:
                    if group["scale_type"] == "channel":
                        numer = torch.norm(norm_grad, dim=norm_dim)
                        denom_scale = torch.norm(low_grad, dim=norm_dim).clamp(min=1e-8)
                        scaling = numer / denom_scale
                        if norm_dim == 1:
                            scaling = scaling.unsqueeze(1)
                    else:
                        numer = torch.norm(norm_grad)
                        denom_scale = torch.norm(low_grad).clamp(min=1e-8)
                        scaling = numer / denom_scale

                    update = grad * scaling
                    scale_coeff = math.sqrt(max(float(group["scale"]), 0.0))
                    if group["scale_front"]:
                        update = update * scale_coeff

                    if not group["disable_nl"]:
                        update_norm = torch.norm(update)
                        prev_norm = state.get("scaled_grad_norm")
                        if prev_norm is not None:
                            limiter = (
                                max(float(update_norm / (prev_norm + EPS)), NORM_LIMITER_THRESHOLD)
                                / NORM_LIMITER_THRESHOLD
                            )
                            update = update / limiter
                            state["scaled_grad_norm"] = update_norm / limiter
                        else:
                            state["scaled_grad_norm"] = update_norm

                    if not group["scale_front"]:
                        update = update * scale_coeff
                else:
                    update = norm_grad

                if wd > 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-step_size)

        return loss
