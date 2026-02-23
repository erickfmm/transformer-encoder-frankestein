from __future__ import annotations

import torch
from torch.optim import AdamW


class CautiousAdamW(AdamW):
    """
    Cautious AdamW: clips gradients before the base AdamW step.
    """

    def __init__(self, params, *args, cautious_clip=1.0, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.cautious_clip = float(cautious_clip)

    @torch.no_grad()
    def step(self, closure=None):
        if self.cautious_clip > 0:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    p.grad.data.clamp_(-self.cautious_clip, self.cautious_clip)
        return super().step(closure=closure)
