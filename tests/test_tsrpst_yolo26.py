"""CPU smoke tests for the TS-RPST / YOLO26 port (``tsrpst_yolo26``).

Run with pytest or directly:  ``python tests/test_tsrpst_yolo26.py``
"""

from __future__ import annotations

import copy
import io
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tsrpst_yolo26 import (  # noqa: E402
    LocalEMATeacher,
    RPSTLoss,
    TinyObjectSurvivalDownsample,
    TSRPSTDetectionModel,
    TSRPSTState,
    TSRPSTTrainer,
    UAFeatureUpsample,
    build_survival_target,
    load_tsrpst_hyp,
)


def _fast_hyp():
    hyp = load_tsrpst_hyp()
    hyp.update(rpst_start_epoch=0, rpst_warmup_epochs=1, rpst_interval=1, rpst_crop_size=64)
    return hyp


def _synthetic_batch():
    torch.manual_seed(0)
    imgs = torch.rand(2, 3, 128, 128)
    cls = torch.tensor([[0], [1], [2], [0], [1], [2], [0]]).float()
    batch_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1]).float()
    bboxes = torch.tensor(
        [
            [0.20, 0.20, 0.05, 0.05],
            [0.50, 0.50, 0.08, 0.06],
            [0.80, 0.30, 0.03, 0.04],
            [0.30, 0.70, 0.06, 0.05],
            [0.60, 0.60, 0.04, 0.04],
            [0.70, 0.20, 0.10, 0.10],
            [0.50, 0.50, 0.60, 0.60],  # large object: never selected for RPST
        ]
    )
    return dict(img=imgs, cls=cls, batch_idx=batch_idx, bboxes=bboxes)


def _make_dataset(root: Path, n_train=16, n_val=4, size=160):
    import cv2
    import yaml

    random.seed(0)
    np.random.seed(0)
    for split, n in (("train", n_train), ("val", n_val)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = np.full((size, size, 3), 90, np.uint8) + np.random.randint(0, 30, (size, size, 3), np.uint8)
            lines = []
            for _ in range(random.randint(3, 6)):
                w, h = random.randint(3, 14), random.randint(3, 14)
                x, y = random.randint(w, size - w), random.randint(h, size - h)
                c = random.randint(0, 2)
                cv2.rectangle(img, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (255, 255, 255), -1)
                lines.append(f"{c} {x / size:.6f} {y / size:.6f} {w / size:.6f} {h / size:.6f}")
            cv2.imwrite(str(root / "images" / split / f"{i:03d}.png"), img)
            (root / "labels" / split / f"{i:03d}.txt").write_text("\n".join(lines) + "\n")
    data_yaml = root / "synth.yaml"
    data_yaml.write_text(
        yaml.safe_dump(dict(path=str(root), train="images/train", val="images/val", names={0: "a", 1: "b", 2: "c"}))
    )
    return data_yaml


# --------------------------------------------------------------------------------------------------- tests
def test_build_variants():
    expectations = {
        "yolo26n-tsrpst.yaml": dict(tosd=[1, 3], ua=[11, 14], strides=[8.0, 16.0, 32.0]),
        "yolo26n-tsrpst-p2.yaml": dict(tosd=[1, 3], ua=[11, 14, 17], strides=[4.0, 8.0, 16.0, 32.0]),
    }
    for cfg, exp in expectations.items():
        model = TSRPSTDetectionModel(cfg, nc=3, verbose=False)
        tosd = [i for i, m in enumerate(model.model) if isinstance(m, TinyObjectSurvivalDownsample)]
        ua = [i for i, m in enumerate(model.model) if isinstance(m, UAFeatureUpsample)]
        assert tosd == exp["tosd"], (cfg, tosd)
        assert ua == exp["ua"], (cfg, ua)
        assert model.stride.tolist() == exp["strides"], (cfg, model.stride)
        assert model.yaml["scale"] == "n"
        assert any(row[2] == "TinyObjectSurvivalDownsample" for row in model.yaml["backbone"])
        for i in ua:  # guide layers must be kept in the save list
            assert all(f in model.save for f in model.model[i].f if f != -1)

        model.eval()
        with torch.no_grad():
            decoded, raw = model(torch.zeros(1, 3, 128, 128))
        assert decoded.shape[0] == 1
        assert len(raw["one2many"]["feats"]) == len(exp["strides"])

        model.train()
        out = model(torch.zeros(2, 3, 128, 128))
        assert set(out) == {"one2many", "one2one"}
        assert out["one2many"]["feats"][0].shape[0] == 2


def test_survival_target_prefers_tiny_objects():
    logits = torch.zeros(1, 1, 32, 32)
    targets = torch.tensor([[0, 0, 0.25, 0.25, 0.05, 0.05], [0, 0, 0.75, 0.75, 0.5, 0.5]])
    target = build_survival_target(logits, targets, tau=4.0)
    assert target.shape == (1, 1, 32, 32)
    tiny_peak = target[0, 0, 7:9, 7:9].max()
    big_peak = target[0, 0, 22:26, 22:26].max()
    # 1.6x1.6-cell object: r = exp(-2.56 / 4) = 0.53, sigma clamped to 1 -> peak ~0.41 half a cell off-centre
    assert 0.3 < tiny_peak < 0.6
    assert big_peak < 1e-3  # exp(-area / tau) vanishes for a 16x16-cell object


def test_loss_with_survival_and_rpst():
    model = TSRPSTDetectionModel("yolo26n-tsrpst-p2.yaml", nc=3, verbose=False)
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=10)
    hyp = _fast_hyp()
    model.tsrpst_hyp = hyp
    teacher = LocalEMATeacher(model, decay=0.999)
    model.attach_tsrpst_state(TSRPSTState(teacher, RPSTLoss(hyp), epoch=0))
    batch = _synthetic_batch()

    model.train()
    loss, items = model.loss(batch)
    assert torch.isfinite(loss)
    assert {"box_loss", "cls_loss", "surv_loss", "rpst_loss"} <= set(items)
    assert items["surv_loss"] > 0
    assert items["rpst_loss"] > 0
    stats = model._tsrpst_state.last_stats
    assert stats["selected_objects"] == 2 * hyp["rpst_max_objects_per_image"]  # 2 per image, big box excluded
    assert stats["effective_weight"] == hyp["rpst_loss_weight"]
    loss.backward()
    assert model.model[1].gamma.grad is not None and torch.isfinite(model.model[1].gamma.grad)
    assert model.model[11].gamma.grad is not None
    assert model.model[1].last_risk_logits is None  # transient activations cleared after loss

    # validation / inference mode: auxiliary terms are zero but the keys stay (validator accumulates by key)
    model.eval()
    with torch.no_grad():
        _, eval_items = model.loss(batch)
    assert float(eval_items["surv_loss"]) == 0.0 and float(eval_items["rpst_loss"]) == 0.0

    # EMA / checkpoint copies never carry the teacher
    model.train()
    model.loss(batch)
    clone = copy.deepcopy(model)
    assert clone._tsrpst_state is None and model._tsrpst_state.teacher is teacher
    buffer = io.BytesIO()
    torch.save({"model": clone}, buffer)
    buffer.seek(0)
    reloaded = torch.load(buffer, weights_only=False)["model"]
    assert isinstance(reloaded, TSRPSTDetectionModel)

    # teacher EMA update moves towards the student
    with torch.no_grad():
        model.model[1].gamma.fill_(1.0)
    teacher.update(model)
    assert teacher.updates == 1
    assert abs(float(teacher.ema.model[1].gamma) - 0.001) < 1e-6


def test_rpst_schedule_gating():
    hyp = load_tsrpst_hyp()
    loss_fn = RPSTLoss(hyp)
    assert loss_fn.effective_weight(hyp["rpst_start_epoch"] - 1) == 0.0
    assert 0 < loss_fn.effective_weight(hyp["rpst_start_epoch"]) < hyp["rpst_loss_weight"]
    assert loss_fn.effective_weight(hyp["rpst_start_epoch"] + hyp["rpst_warmup_epochs"]) == hyp["rpst_loss_weight"]
    imgs = torch.zeros(1, 3, 64, 64)
    zero, stats = loss_fn([torch.zeros(1, 8, 16, 16)], None, imgs, None, 100, 0)
    assert float(zero) == 0.0 and float(stats["selected_objects"]) == 0.0
    zero, _ = loss_fn([torch.zeros(1, 8, 16, 16)], object(), imgs, None, 100, 1)  # off-interval step
    assert float(zero) == 0.0


def test_trainer_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_yaml = _make_dataset(root)
        trainer = TSRPSTTrainer(
            overrides=dict(
                model="yolo26n-tsrpst-p2.yaml",
                data=str(data_yaml),
                epochs=2,
                batch=4,
                nbs=4,  # optimizer step every batch so the RPST teacher is updated
                imgsz=128,
                device="cpu",
                workers=0,
                plots=False,
                amp=False,
                warmup_epochs=0,
                project=str(root / "runs"),
                name="smoke",
                exist_ok=True,
            ),
            tsrpst=_fast_hyp(),
        )
        trainer.train()
        assert trainer.loss_names == ("box_loss", "cls_loss", "l1_loss", "surv_loss", "rpst_loss")
        assert float(trainer.tloss["surv_loss"]) > 0
        assert float(trainer.tloss["rpst_loss"]) > 0
        assert trainer.rpst_teacher.updates > 0
        assert trainer.model._tsrpst_state.epoch == 1
        weights = Path(trainer.save_dir) / "weights" / "best.pt"
        assert weights.exists()

        from ultralytics import YOLO

        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        assert isinstance(ckpt["model"], TSRPSTDetectionModel)
        rebuilt = TSRPSTDetectionModel(ckpt["model"].yaml, nc=3, verbose=False)
        rebuilt.load(ckpt["model"].float())
        yolo = YOLO(str(weights))
        result = yolo.predict(str(root / "images" / "val" / "000.png"), imgsz=128, conf=0.001, device="cpu", verbose=False)
        assert result[0].boxes.shape[1] == 6


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failures else 0)
