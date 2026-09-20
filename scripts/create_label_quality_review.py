from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from green_roofs.label_qa import write_review_package  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create side-by-side images for manual green-roof label review."
    )
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("processed_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = write_review_package(
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        output_root=args.output_root,
        sample_count=args.sample_count,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
