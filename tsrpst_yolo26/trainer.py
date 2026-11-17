"""Ultralytics ``DetectionTrainer`` subclass that trains ``TSRPSTDetectionModel`` with survival + RPST losses.

The TS-RPST hyper-parameters live in a separate YAML (``tsrpst_yolo26/cfg/tsrpst.yaml``) because Ultralytics rejects
unknown keys in its own config. Pass a path or dict as ``tsrpst=`` when constructing the trainer.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, YAML
from ultralytics.utils.torch_utils import unwrap_model

from .model import CFG_DIR, TSRPSTDetectionModel, TSRPSTState
from .rpst import LocalEMATeacher, RPSTLoss

DEFAULT_TSRPST_HYP = CFG_DIR / "tsrpst.yaml"


def load_tsrpst_hyp(src=None) -> dict:
    """Load TS-RPST auxiliary-loss hyper-parameters from a YAML path or dict (default: ``cfg/tsrpst.yaml``)."""
    if src is None:
        src = DEFAULT_TSRPST_HYP
    if isinstance(src, dict):
        return deepcopy(src)
    return dict(YAML.load(Path(src)))


def _sync_epoch(trainer):
    state = getattr(unwrap_model(trainer.model), "_tsrpst_state", None)
    if state is not None:
        state.epoch = trainer.epoch


class TSRPSTTrainer(DetectionTrainer):
    """Detection trainer with TOSD survival supervision and RPST self-distillation."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, tsrpst=None):
        self.tsrpst_hyp = load_tsrpst_hyp(tsrpst)
        super().__init__(cfg, overrides, _callbacks)
        self.rpst_teacher = None
        self.add_callback("on_train_epoch_start", _sync_epoch)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(
            TSRPSTDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        if weights:
            model.load(weights)
        model.tsrpst_hyp = self.tsrpst_hyp
        return model

    def _setup_train(self):
        super()._setup_train()
        base = unwrap_model(self.model)
        base.tsrpst_hyp = self.tsrpst_hyp
        hyp = self.tsrpst_hyp
        if hyp.get("rpst_enabled", True):
            self.rpst_teacher = LocalEMATeacher(base, decay=float(hyp.get("rpst_ema_decay", 0.999)))
            if RANK in {-1, 0}:
                LOGGER.info(
                    f"RPST enabled: EMA teacher (decay={hyp.get('rpst_ema_decay', 0.999)}), "
                    f"start_epoch={hyp.get('rpst_start_epoch', 10)}, warmup={hyp.get('rpst_warmup_epochs', 10)}, "
                    f"interval={hyp.get('rpst_interval', 4)}, weight={hyp.get('rpst_loss_weight', 0.2)}"
                )
        else:
            self.rpst_teacher = None
        if RANK in {-1, 0}:
            LOGGER.info(
                f"TOSD survival loss {'enabled' if hyp.get('survival_enabled', True) else 'disabled'} "
                f"(weight={hyp.get('survival_loss_weight', 0.25)}, tau={hyp.get('survival_tau', 4.0)})"
            )
        base.attach_tsrpst_state(TSRPSTState(self.rpst_teacher, RPSTLoss(hyp), epoch=self.start_epoch))

    def optimizer_step(self):
        super().optimizer_step()
        if self.rpst_teacher is not None:
            self.rpst_teacher.update(unwrap_model(self.model))
