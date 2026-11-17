"""TS-RPST on Ultralytics YOLO26.

Port of the YOLOv5/FFCA-YOLO implementation (``models/``, ``utils/rpst.py``, ``utils/survival_loss.py``) to the
Ultralytics framework (tested against ``ultralytics==8.4.172``):

* :mod:`tsrpst_yolo26.modules`  - ``TinyObjectSurvivalDownsample`` (TOSD) and ``UAFeatureUpsample``
* :mod:`tsrpst_yolo26.survival` - survival-risk supervision for TOSD
* :mod:`tsrpst_yolo26.rpst`     - training-only Resolution-Privileged Scale Transfer
* :mod:`tsrpst_yolo26.model`    - ``TSRPSTDetectionModel`` (YOLO26 + TS-RPST layers + aux losses)
* :mod:`tsrpst_yolo26.trainer`  - ``TSRPSTTrainer``
* ``cfg/``                       - model YAMLs (``yolo26-tsrpst.yaml``, ``yolo26-tsrpst-p2.yaml``) and ``tsrpst.yaml``

Example::

    from tsrpst_yolo26 import TSRPSTTrainer
    TSRPSTTrainer(overrides=dict(model="yolo26s-tsrpst-p2.yaml", data="data/aitod_yolo26.yaml",
                                 epochs=300, batch=32, imgsz=640)).train()
"""

from .model import CFG_DIR, TSRPSTDetectionModel, TSRPSTState, load_model_spec
from .modules import TinyObjectSurvivalDownsample, UAFeatureUpsample
from .rpst import LocalEMATeacher, RPSTLoss, extract_pyramid_features
from .survival import build_survival_target, compute_survival_loss, targets_from_batch
from .trainer import TSRPSTTrainer, load_tsrpst_hyp

__all__ = [
    "CFG_DIR",
    "LocalEMATeacher",
    "RPSTLoss",
    "TSRPSTDetectionModel",
    "TSRPSTState",
    "TSRPSTTrainer",
    "TinyObjectSurvivalDownsample",
    "UAFeatureUpsample",
    "build_survival_target",
    "compute_survival_loss",
    "extract_pyramid_features",
    "load_model_spec",
    "load_tsrpst_hyp",
    "targets_from_batch",
]
