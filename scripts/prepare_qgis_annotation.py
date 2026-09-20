"""Create a georeferenced mosaic and editable labels for QGIS review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


from green_roofs.qgis_annotation import prepare_qgis_annotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("processed_root", type=Path)
    parser.add_argument("output_root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_qgis_annotation(
        args.dataset_root.resolve(),
        args.processed_root.resolve(),
        args.output_root.resolve(),
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
