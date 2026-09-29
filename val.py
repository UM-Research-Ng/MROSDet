"""Evaluate both MROSDet branches on a paired dataset split."""

import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "configs/umod_sample.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for key in ("weights", "data", "output"):
        setattr(args, key, (ROOT / getattr(args, key).expanduser()).resolve())
    if min(args.imgsz, args.batch, args.max_det) < 1 or args.imgsz % 32 or args.workers < 0:
        parser.error("Use a positive batch/max-det, nonnegative workers and imgsz divisible by 32")
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        parser.error("conf and iou must be within [0, 1]")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from mrosdet.engine import validate
    return validate(args)


if __name__ == "__main__":
    main()
