from __future__ import annotations

import warnings

import torch

from .apollo import Apollo

# Keep quantization meaningful (>=2 bits) and compatible with uint8 storage (<=8 bits).
MIN_QUANT_BITS = 2
MAX_QUANT_BITS = 8
MIN_SCALE_EPSILON = 1e-8


class QApollo(Apollo):
    """
    Quantized APOLLO: stores first/second moments in quantized form.
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
        quant_bits=8,
    ):
        super().__init__(
            params=params,
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
        try:
            bits = int(quant_bits)
        except (TypeError, ValueError):
            warnings.warn(
                f"Invalid quant_bits={quant_bits!r}; falling back to {MAX_QUANT_BITS}.",
                RuntimeWarning,
            )
            bits = MAX_QUANT_BITS
        self.quant_bits = max(MIN_QUANT_BITS, min(bits, MAX_QUANT_BITS))

    def _quantize(self, value):
        max_int = float((1 << self.quant_bits) - 1)
        v_min = value.min()
        v_max = value.max()
        scale = (v_max - v_min).clamp(min=MIN_SCALE_EPSILON) / max_int
        q = torch.clamp(torch.round((value - v_min) / scale), 0, max_int).to(torch.uint8)
        return q, scale, v_min

    @staticmethod
    def _dequantize(q_value, scale, v_min, dtype):
        return q_value.to(dtype=dtype) * scale + v_min

    def _load_moments(self, state, grad_like):
        if "exp_avg_q" in state and "exp_avg_sq_q" in state:
            exp_avg = self._dequantize(
                state["exp_avg_q"],
                state["exp_avg_scale"],
                state["exp_avg_min"],
                grad_like.dtype,
            )
            exp_avg_sq = self._dequantize(
                state["exp_avg_sq_q"],
                state["exp_avg_sq_scale"],
                state["exp_avg_sq_min"],
                grad_like.dtype,
            )
            if exp_avg.shape == grad_like.shape and exp_avg_sq.shape == grad_like.shape:
                return exp_avg, exp_avg_sq

        return torch.zeros_like(grad_like), torch.zeros_like(grad_like)

    def _store_moments(self, state, exp_avg, exp_avg_sq):
        q_avg, s_avg, m_avg = self._quantize(exp_avg)
        q_avg_sq, s_avg_sq, m_avg_sq = self._quantize(exp_avg_sq)
        state["exp_avg_q"] = q_avg
        state["exp_avg_scale"] = s_avg
        state["exp_avg_min"] = m_avg
        state["exp_avg_sq_q"] = q_avg_sq
        state["exp_avg_sq_scale"] = s_avg_sq
        state["exp_avg_sq_min"] = m_avg_sq
        state.pop("exp_avg", None)
        state.pop("exp_avg_sq", None)
