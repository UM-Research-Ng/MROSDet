"""训练 MROSDet；默认随机初始化，指定 --weights 时加载权重进行微调。"""

import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "--config", dest="config", type=Path, default=ROOT / "configs/mrosdet.yaml")
    parser.add_argument("--data", type=Path, default=ROOT / "configs/umod_sample.yaml")
    parser.add_argument("--weights", type=Path, help="Optional checkpoint to fine-tune; omitted means random initialization.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/train")
    parser.add_argument("--device", default="auto", help="auto, cpu, or one CUDA device, e.g. 0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr0", type=float, default=0.000769)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--nbs", type=int, default=64, help="Nominal batch size for accumulation and weight-decay scaling.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for key in ("config", "data", "weights", "output"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, (ROOT / value.expanduser()).resolve())
    if min(args.epochs, args.batch, args.nbs, args.imgsz) < 1 or args.imgsz % 32:
        parser.error("epochs, batch and nbs must be positive; imgsz must be a positive multiple of 32")
    if args.workers < 0 or args.warmup_epochs < 0 or args.lr0 <= 0 or not 0 < args.lrf <= 1 or args.weight_decay < 0:
        parser.error("Invalid workers, warmup, learning rate, or weight decay")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from mrosdet.engine import train
    return train(args)


if __name__ == "__main__":
    main()
