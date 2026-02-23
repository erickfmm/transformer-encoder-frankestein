from __future__ import annotations

from torch.optim import AdamW


class Adan(AdamW):
    """
    Practical approximation backed by AdamW internals for compatibility.
    """

    def __init__(self, params, lr=1e-3, betas=(0.98, 0.92), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
