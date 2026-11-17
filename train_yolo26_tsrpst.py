"""Train TS-RPST on Ultralytics YOLO26.

Examples (AI-TOD, paper protocol: 640 px, batch 32, 300 epochs):

    python train_yolo26_tsrpst.py --model yolo26s-tsrpst-p2.yaml --data data/aitod_yolo26.yaml \
        --epochs 300 --batch 32 --imgsz 640 --device 0 --name yolo26s_tsrpst_p2

Ablations:

    --no-survival         TOSD without survival supervision (layers stay, risk gate is unsupervised)
    --no-rpst             disable RPST (no EMA teacher, no distillation)
    --model yolo26s-p2.yaml    plain YOLO26 baseline (no TOSD / UA layers; aux losses are then zero)

Any other Ultralytics train argument can be passed as `--opt key=value`, e.g. `--opt optimizer=MuSGD cos_lr=True`.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tsrpst_yolo26 import TSRPSTTrainer, load_tsrpst_hyp  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolo26s-tsrpst-p2.yaml", help="model yaml (yolo26{n,s,m,l,x}-tsrpst[-p2].yaml) or .pt")
    p.add_argument("--data", default=str(ROOT / "data" / "aitod_yolo26.yaml"), help="dataset yaml")
    p.add_argument("--tsrpst", default=None, help="TS-RPST hyper-parameter yaml (default: tsrpst_yolo26/cfg/tsrpst.yaml)")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="", help="cuda device id(s), 'cpu', or '' for auto")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", default=str(ROOT / "runs" / "tsrpst_yolo26"))
    p.add_argument("--name", default=None)
    p.add_argument("--optimizer", default="auto", help="auto (MuSGD for long runs), MuSGD, SGD, AdamW, ...")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-survival", action="store_true", help="disable TOSD survival loss")
    p.add_argument("--no-rpst", action="store_true", help="disable RPST distillation")
    p.add_argument("--opt", nargs="*", default=[], metavar="KEY=VALUE", help="extra Ultralytics train arguments")
    return p.parse_args(argv)


def _literal(value: str):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def main(argv=None):
    args = parse_args(argv)
    hyp = load_tsrpst_hyp(args.tsrpst)
    if args.no_survival:
        hyp["survival_enabled"] = False
    if args.no_rpst:
        hyp["rpst_enabled"] = False

    overrides = dict(
        model=args.model,
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        project=args.project,
        optimizer=args.optimizer,
        resume=args.resume,
    )
    if args.device != "":
        overrides["device"] = args.device
    if args.name:
        overrides["name"] = args.name
    for item in args.opt:
        if "=" not in item:
            raise SystemExit(f"--opt expects KEY=VALUE, got '{item}'")
        key, value = item.split("=", 1)
        overrides[key.strip()] = _literal(value.strip())

    trainer = TSRPSTTrainer(overrides=overrides, tsrpst=hyp)
    trainer.train()
    return trainer


if __name__ == "__main__":
    main()
