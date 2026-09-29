"""Predict paired optical/sonar images and save per-modality images and JSON boxes."""

import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "configs/umod_sample.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--rgb", type=Path, help="RGB image directory; requires --sonar")
    parser.add_argument("--sonar", type=Path, help="Sonar image directory; requires --rgb")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/predict")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0, help="Maximum image pairs; 0 means all pairs")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for key in ("weights", "data", "rgb", "sonar", "output"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, (ROOT / value.expanduser()).resolve())
    if bool(args.rgb) != bool(args.sonar):
        parser.error("--rgb and --sonar must be supplied together")
    if args.imgsz < 32 or args.imgsz % 32 or args.max_det < 1 or args.limit < 0:
        parser.error("Use imgsz divisible by 32, positive max-det and nonnegative limit")
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        parser.error("conf and iou must be within [0, 1]")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from mrosdet.engine import predict
    return predict(args)


if __name__ == "__main__":
    main()
