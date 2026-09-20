"""Validate a prepared green-roof segmentation dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from green_roofs.preprocessing import validate_prepared_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    result = validate_prepared_dataset(args.dataset_root, args.output_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

