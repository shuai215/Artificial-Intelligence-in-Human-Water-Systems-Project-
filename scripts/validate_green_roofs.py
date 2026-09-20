"""Validate a prepared green-roof segmentation dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


from green_roofs.preprocessing import validate_prepared_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    result = validate_prepared_dataset(args.dataset_root, args.output_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
