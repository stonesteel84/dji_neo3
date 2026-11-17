"""YOLO26 detection model with TS-RPST modules and auxiliary losses.

Ultralytics' ``parse_model`` only knows its own layer classes, so a YAML that names ``TinyObjectSurvivalDownsample``
or ``UAFeatureUpsample`` is first rewritten with shape-equivalent placeholders (``Conv`` and ``nn.Upsample``), built by
the stock parser, and then the placeholders are swapped for the real modules. This keeps width/depth scaling, stride
inference, and checkpoint handling identical to a plain YOLO26 model.

``TSRPSTDetectionModel.loss`` adds the survival loss (TOSD) and the RPST distillation loss to the YOLO26 detection
loss during training; at validation and inference it behaves exactly like ``DetectionModel``.
"""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.tasks import DetectionModel, guess_model_scale, yaml_model_load
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import initialize_weights

from .modules import TinyObjectSurvivalDownsample, UAFeatureUpsample
from .rpst import RPSTLoss, extract_pyramid_features
from .survival import compute_survival_loss, survival_modules, targets_from_batch

CFG_DIR = Path(__file__).resolve().parent / "cfg"
TOSD_NAME = "TinyObjectSurvivalDownsample"
UA_NAME = "UAFeatureUpsample"
CUSTOM_MODULES = {TOSD_NAME: TinyObjectSurvivalDownsample, UA_NAME: UAFeatureUpsample}


def load_model_spec(cfg) -> dict:
    """Load a model YAML into a dict, resolving scaled names (``yolo26s-tsrpst.yaml``) against ``tsrpst_yolo26/cfg``."""
    if isinstance(cfg, dict):
        return deepcopy(cfg)
    path = Path(cfg)
    if path.exists():
        return yaml_model_load(path)
    unified = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", path.name)  # yolo26s-tsrpst.yaml -> yolo26-tsrpst.yaml
    for candidate in (CFG_DIR / path.name, CFG_DIR / unified):
        if candidate.exists():
            spec = yaml_model_load(candidate)
            spec["scale"] = guess_model_scale(path) or spec.get("scale", "")
            spec["yaml_file"] = str(path)
            return spec
    raise FileNotFoundError(f"Model config '{cfg}' not found locally or in {CFG_DIR}")


def to_placeholder_spec(spec: dict):
    """Replace TS-RPST modules by stock modules with the same channel/stride behaviour.

    Returns ``(placeholder_spec, custom)`` where ``custom`` maps layer index -> ``(name, from, args)``.
    """
    spec = deepcopy(spec)
    nb = len(spec["backbone"])
    custom = {}
    for i, (f, n, m, args) in enumerate(spec["backbone"] + spec["head"]):
        if m == TOSD_NAME:
            c2 = args[0]
            k = args[1] if len(args) > 1 else 3
            s = args[2] if len(args) > 2 else 2
            custom[i] = (m, f, list(args))
            new = [f, n, "Conv", [c2, k, s]]
        elif m == UA_NAME:
            if not isinstance(f, (list, tuple)) or len(f) != 2:
                raise ValueError(f"{UA_NAME} at layer {i} needs from=[x_low, x_guide], got {f}")
            scale = args[3] if len(args) > 3 else 2
            custom[i] = (m, list(f), list(args))
            new = [f[0], n, "nn.Upsample", [None, scale, "nearest"]]
        else:
            continue
        if i < nb:
            spec["backbone"][i] = new
        else:
            spec["head"][i - nb] = new
    return spec, custom


class TSRPSTState:
    """Training-only RPST context attached to the student model by the trainer (never pickled)."""

    def __init__(self, teacher=None, loss_fn: RPSTLoss | None = None, epoch: int = 0):
        self.teacher = teacher
        self.loss_fn = loss_fn
        self.epoch = int(epoch)
        self.step = 0
        self.last_stats = None


class TSRPSTDetectionModel(DetectionModel):
    """YOLO26 ``DetectionModel`` with TOSD / UAFeatureUpsample layers and survival + RPST auxiliary losses."""

    def __init__(self, cfg="yolo26s-tsrpst.yaml", ch=3, nc=None, verbose=True):
        spec = load_model_spec(cfg)
        placeholder_spec, custom = to_placeholder_spec(spec)
        super().__init__(placeholder_spec, ch=ch, nc=nc, verbose=False)
        self.tsrpst_hyp: dict = {}
        self._tsrpst_state: TSRPSTState | None = None
        if custom:
            self._swap_custom_modules(custom, ch)
            initialize_weights(self)  # BN eps/momentum + inplace act for the new modules
        spec["nc"], spec["channels"] = self.yaml["nc"], self.yaml["channels"]
        spec.setdefault("scale", self.yaml.get("scale", ""))
        self.yaml = spec  # keep the original (TS-RPST) spec so checkpoints rebuild through this class
        if verbose:
            for i, (name, f, args) in sorted(custom.items()):
                LOGGER.info(f"TS-RPST layer {i:>3}: {name}(from={f}, args={args})")
            self.info()
            LOGGER.info("")

    # ------------------------------------------------------------------ construction
    def _layer_out_channels(self, ch: int) -> dict:
        channels = {}
        hooks = []
        for idx, m in enumerate(self.model):
            def _hook(mod, inp, out, idx=idx):
                channels[idx] = int(out.shape[1]) if torch.is_tensor(out) and out.ndim == 4 else None
            hooks.append(m.register_forward_hook(_hook))
        was_training = self.training
        self.eval()
        with torch.no_grad():
            self.forward(torch.zeros(1, ch, 64, 64))
        for h in hooks:
            h.remove()
        self.train(was_training)
        return channels

    def _swap_custom_modules(self, custom: dict, ch: int):
        out_ch = self._layer_out_channels(ch)
        for i, (name, f, args) in sorted(custom.items()):
            old = self.model[i]
            if name == TOSD_NAME:
                extra = list(args[3:])  # risk_channels, gamma_init (optional)
                k = args[1] if len(args) > 1 else 3
                s = args[2] if len(args) > 2 else 2
                new = TinyObjectSurvivalDownsample(old.conv.in_channels, old.conv.out_channels, k, s, *extra)
            elif name == UA_NAME:
                f_low, f_guide = f
                c_low = out_ch[i - 1 if f_low == -1 else f_low]
                c_guide = out_ch[f_guide]
                new = UAFeatureUpsample(c_low, c_guide, *args)
            else:  # pragma: no cover
                raise KeyError(name)
            new.i, new.f, new.type = i, f, f"tsrpst_yolo26.modules.{name}"
            new.np = sum(p.numel() for p in new.parameters())
            self.model[i] = new
            if isinstance(f, (list, tuple)):
                self.save = sorted(set(self.save) | {x % i for x in f if x != -1})

    # ------------------------------------------------------------------ state handling
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tsrpst_state"] = None  # teacher / loss context is training-only
        return state

    def attach_tsrpst_state(self, state: TSRPSTState | None):
        self._tsrpst_state = state

    def clear_transient(self):
        for m in survival_modules(self):
            m.clear_transient()

    # ------------------------------------------------------------------ loss
    def loss(self, batch, preds=None):
        """Detection loss + survival loss (TOSD) + RPST loss. Aux terms are zero outside training mode."""
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward(batch["img"])
        det_loss, items = self.criterion(preds, batch)
        total = det_loss.sum()
        zero = total.new_zeros(())
        surv_loss, rpst_loss = zero, zero

        hyp = self.tsrpst_hyp or {}
        if self.training and hyp:
            imgs = batch["img"]
            targets = targets_from_batch(batch).to(imgs.device)
            if hyp.get("survival_enabled", True):
                surv_loss, _ = compute_survival_loss(self, targets, hyp, reference=total)
            state = self._tsrpst_state
            if state is not None and state.teacher is not None and state.loss_fn is not None and hyp.get("rpst_enabled", True):
                feats = extract_pyramid_features(preds)
                rpst_loss, state.last_stats = state.loss_fn(feats, state.teacher, imgs, targets, state.epoch, state.step)
            if state is not None:
                state.step += 1
            # The YOLOv5 reference implementation adds the auxiliary terms un-scaled on top of the batch-scaled
            # detection loss; keep that by default so lambda_surv / lambda_rpst mean the same thing as in the paper.
            scale = float(imgs.shape[0]) if hyp.get("aux_scale_by_batch", False) else 1.0
            total = total + scale * (float(hyp.get("survival_loss_weight", 0.25)) * surv_loss + rpst_loss)

        items = dict(items)
        items["surv_loss"] = surv_loss.detach()
        items["rpst_loss"] = rpst_loss.detach()
        self.clear_transient()
        return total, items
