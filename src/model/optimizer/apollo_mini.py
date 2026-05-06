from __future__ import annotations

from .apollo import Apollo


class ApolloMini(Apollo):
    """
    APOLLO-Mini: APOLLO configured with rank-1 tensor-wise scaling.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        update_proj_gap=200,
        scale=128.0,
        proj_type="std",
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        correct_bias=True,
        scale_front=False,
        disable_nl=False,
    ):
        super().__init__(
            params=params,
            lr=lr,
            rank=1,
            update_proj_gap=update_proj_gap,
            scale=scale,
            scale_type="tensor",
            proj_type=proj_type,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
            scale_front=scale_front,
            disable_nl=disable_nl,
        )
