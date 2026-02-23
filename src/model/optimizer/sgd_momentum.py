from __future__ import annotations

from torch.optim import SGD


class SGDMomentum(SGD):
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.0, nesterov=False):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            dampening=0.0,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
