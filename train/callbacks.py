"""Lightning callbacks that instrument a training run for wandb.

Only `train.cli` attaches these; the cross-validation path in `evaluate.cli` runs
without a logger and without them. They are deliberately callbacks rather than
inline setup because `train_func` swaps `self.model` for the best checkpoint once
`fit` returns — the module that actually gets trained is the `pl_module` Lightning
hands to these hooks, and that is the one whose gradients we want.
"""

import lightning as L
from lightning.pytorch.utilities import grad_norm


class GradNormLogger(L.Callback):
    """Log per-parameter and total gradient L2 norms on each optimizer step.

    Gradients are read in `on_before_optimizer_step`, i.e. after backward and before
    the update, so these are the raw norms the optimizer is about to consume. The
    chameleon backbone contributes a few hundred keys per step, so `every_n_steps`
    is there to thin that out on longer runs.
    """

    def __init__(self, norm_type: float = 2.0, every_n_steps: int = 1) -> None:
        super().__init__()
        self.norm_type = norm_type
        self.every_n_steps = every_n_steps

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if self.every_n_steps > 1 and trainer.global_step % self.every_n_steps:
            return

        pl_module.log_dict(grad_norm(pl_module, norm_type=self.norm_type))


class WandbWatch(L.Callback):
    """Log parameter/gradient histograms for the module being fit, via `wandb.watch`.

    `log_graph=False` on purpose: the graph tracer cannot walk chemprop's
    `BatchMolGraph` inputs. The hooks `watch` installs are removed in `on_fit_end`
    so they don't fire during the later `predict_func` passes.

    The mode is held as `log_mode`, not `log`: Lightning binds the LightningModule's
    `log` method onto every attached callback, so an attribute named `log` is
    silently replaced by that method before the first hook runs.
    """

    def __init__(self, log_mode: str = "all", log_freq: int = 50) -> None:
        super().__init__()
        self.log_mode = log_mode
        self.log_freq = log_freq

    def on_fit_start(self, trainer, pl_module):
        trainer.logger.watch(  # type: ignore[union-attr]
            pl_module,
            log=self.log_mode,
            log_freq=self.log_freq,
            log_graph=False,
        )

    def on_fit_end(self, trainer, pl_module):
        # Go through the logger's run rather than the `wandb.unwatch` module global,
        # which is only rebound to the active run as a side effect of `wandb.init`.
        trainer.logger.experiment.unwatch(pl_module)  # type: ignore[union-attr]
