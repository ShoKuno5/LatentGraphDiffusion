"""Lightning helpers for running modules without a Trainer."""

from __future__ import annotations

import pytorch_lightning as pl


class OptionalTrainerLightningModule(pl.LightningModule):
    """LightningModule that tolerates Trainer-less usage.

    GraphGym drives our diffusion/flow modules directly, so ``self.trainer``
    remains ``None`` and vanilla ``self.log`` would warn/raise.  This subclass
    simply no-ops logging until a Trainer attaches, but behaves like the
    standard LightningModule when one is present.
    """

    def log(self, *args, **kwargs):  # type: ignore[override]
        if getattr(self, "_trainer", None) is None:
            return None
        return super().log(*args, **kwargs)

    def log_dict(self, *args, **kwargs):  # type: ignore[override]
        if getattr(self, "_trainer", None) is None:
            return None
        return super().log_dict(*args, **kwargs)


__all__ = ["OptionalTrainerLightningModule"]
