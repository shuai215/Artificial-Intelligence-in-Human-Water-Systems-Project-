"""Generate green-roof masks and spatial train/validation/test manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


from green_roofs.preprocessing import prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--study-area",
        type=Path,
        help="Polygon layer; keep tiles whose centres fall inside it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.block_size < 1:
        raise ValueError("--block-size must be at least 1")
    summary = prepare_dataset(
        args.dataset_root,
        args.output_root,
        block_size=args.block_size,
        seed=args.seed,
        study_area_path=args.study_area,
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
