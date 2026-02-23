from __future__ import annotations

from torch.optim import AdamW


class ScheduleFreeAdamW(AdamW):
    """
    Schedule-free interface: trainer scheduler can be set to constant.
    """

    pass
