"""Compare train-to-train and prediction-to-train spatial distances."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


from green_roofs.spatial_distance import (
    DISTANCE_CRS,
    ManifestTile,
    log_distance_kde,
    nearest_distances,
    read_manifest,
    tile_centres_to_projected,
)


TRAIN_COLOR = "#D9798B"
PREDICTION_COLOR = "#169C95"


def describe(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(len(values)),
        "minimum_m": float(np.min(values)),
        "p10_m": float(np.percentile(values, 10)),
        "median_m": float(np.median(values)),
        "p90_m": float(np.percentile(values, 90)),
        "p95_m": float(np.percentile(values, 95)),
        "maximum_m": float(np.max(values)),
    }


def write_source_data(
    path: Path,
    train_tiles: list[ManifestTile],
    prediction_tiles: list[ManifestTile],
    train_distances: np.ndarray,
    prediction_distances: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["tile_id", "split", "distance_function", "nearest_distance_m"],
        )
        writer.writeheader()
        for tile, distance in zip(train_tiles, train_distances):
            writer.writerow(
                {
                    "tile_id": tile.tile_id,
                    "split": tile.split,
                    "distance_function": "sample-to-sample",
                    "nearest_distance_m": f"{distance:.6f}",
                }
            )
        for tile, distance in zip(prediction_tiles, prediction_distances):
            writer.writerow(
                {
                    "tile_id": tile.tile_id,
                    "split": tile.split,
                    "distance_function": "sample-to-prediction",
                    "nearest_distance_m": f"{distance:.6f}",
                }
            )


def make_figure(
    output_stem: Path,
    train_points: np.ndarray,
    prediction_points: np.ndarray,
    train_distances: np.ndarray,
    prediction_distances: np.ndarray,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure, (map_axis, density_axis) = plt.subplots(
        1,
        2,
        figsize=(7.09, 3.25),
        gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.34},
    )

    all_points = np.vstack((train_points, prediction_points))
    offset = np.min(all_points, axis=0)
    train_km = (train_points - offset) / 1000.0
    prediction_km = (prediction_points - offset) / 1000.0
    map_axis.scatter(
        prediction_km[:, 0],
        prediction_km[:, 1],
        s=7,
        marker="s",
        color=PREDICTION_COLOR,
        alpha=0.68,
        linewidths=0,
        rasterized=True,
        label=f"Held-out prediction tiles (n={len(prediction_points):,})",
    )
    map_axis.scatter(
        train_km[:, 0],
        train_km[:, 1],
        s=6,
        marker="o",
        color=TRAIN_COLOR,
        alpha=0.82,
        linewidths=0,
        rasterized=True,
        label=f"Training tiles (n={len(train_points):,})",
    )
    map_axis.set_aspect("equal")
    map_axis.set_xlabel("Distance east (km)")
    map_axis.set_ylabel("Distance north (km)")
    map_axis.set_title("a  Spatial train–prediction layout", loc="left", fontweight="bold")
    map_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        handletextpad=0.5,
        borderaxespad=0,
        ncol=1,
    )

    combined = np.concatenate((train_distances, prediction_distances))
    if np.any(~np.isfinite(combined)) or np.any(combined <= 0):
        raise ValueError("Log-scale distances must be finite and strictly positive")
    padding = 0.13
    log_min = np.log10(np.min(combined)) - padding
    log_max = np.log10(np.max(combined)) + padding
    log_grid = np.linspace(log_min, log_max, 600)
    distance_grid = 10**log_grid
    train_density = log_distance_kde(train_distances, log_grid)
    prediction_density = log_distance_kde(prediction_distances, log_grid)
    density_axis.plot(
        distance_grid,
        train_density,
        color=TRAIN_COLOR,
        linewidth=1.8,
        label="Sample-to-sample",
    )
    density_axis.fill_between(distance_grid, train_density, color=TRAIN_COLOR, alpha=0.12)
    density_axis.plot(
        distance_grid,
        prediction_density,
        color=PREDICTION_COLOR,
        linewidth=1.8,
        label="Sample-to-prediction",
    )
    density_axis.fill_between(
        distance_grid, prediction_density, color=PREDICTION_COLOR, alpha=0.12
    )
    density_axis.set_xscale("log")
    density_axis.set_xlim(distance_grid[0], distance_grid[-1])
    candidate_ticks = np.array([10, 20, 50, 100, 200, 500, 1000, 2000], dtype=float)
    visible_ticks = candidate_ticks[
        (candidate_ticks >= distance_grid[0]) & (candidate_ticks <= distance_grid[-1])
    ]
    density_axis.set_xticks(visible_ticks)
    density_axis.set_xticklabels([f"{tick:.0f}" for tick in visible_ticks])
    density_axis.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    density_axis.set_ylim(bottom=0)
    density_axis.set_xlabel("Nearest-neighbour distance (m, log scale)")
    density_axis.set_ylabel("Density of log10(distance)")
    density_axis.set_title("b  Spatial distance distributions", loc="left", fontweight="bold")
    density_axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.7)
    density_axis.legend(loc="upper right")

    train_median = float(np.median(train_distances))
    prediction_median = float(np.median(prediction_distances))
    density_axis.axvline(train_median, color=TRAIN_COLOR, linestyle="--", linewidth=0.8)
    density_axis.axvline(
        prediction_median, color=PREDICTION_COLOR, linestyle="--", linewidth=0.8
    )
    density_axis.text(
        0.97,
        0.66,
        f"Median distance\n{train_median:.0f} m sample; "
        f"{prediction_median:.0f} m prediction",
        color="#333333",
        ha="right",
        va="top",
        transform=density_axis.transAxes,
        fontsize=6,
    )

    figure.text(
        0.01,
        0.01,
        "Study area: Berlin postal code 12357; distances are between XYZ tile centres. "
        "Prediction tiles combine the frozen validation and test splits.",
        fontsize=5.4,
        color="#4D4D4D",
    )
    figure.subplots_adjust(bottom=0.28, top=0.91, left=0.08, right=0.98)
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()

    processed_root = arguments.processed_root.resolve()
    output_dir = (
        arguments.output_dir.resolve()
        if arguments.output_dir
        else processed_root / "spatial_distance_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_tiles = read_manifest(processed_root / "manifests" / "train.csv", "train")
    validation_tiles = read_manifest(processed_root / "manifests" / "val.csv", "val")
    test_tiles = read_manifest(processed_root / "manifests" / "test.csv", "test")
    prediction_tiles = validation_tiles + test_tiles
    all_tiles = train_tiles + prediction_tiles
    tile_xy = np.array([(tile.x, tile.y) for tile in all_tiles], dtype=np.float64)
    local_points = tile_centres_to_projected(tile_xy)
    train_points = local_points[: len(train_tiles)]
    prediction_points = local_points[len(train_tiles) :]

    train_distances = nearest_distances(
        train_points, train_points, exclude_matching_index=True
    )
    prediction_distances = nearest_distances(prediction_points, train_points)
    train_summary = describe(train_distances)
    prediction_summary = describe(prediction_distances)
    summary = {
        "study_area": "Berlin postal code 12357 (not the full Berlin administrative area)",
        "coordinate_method": {
            "tile_locations": "centres of XYZ zoom-19 tiles",
            "distance_crs": DISTANCE_CRS,
            "projection": "ETRS89 / UTM zone 33N via Pyproj",
            "unit": "metre",
        },
        "definitions": {
            "sample-to-sample": (
                "For each frozen training tile, distance to the nearest other training tile."
            ),
            "sample-to-prediction": (
                "For each frozen validation or test tile, distance to the nearest training tile."
            ),
        },
        "sample_to_sample": train_summary,
        "sample_to_prediction": prediction_summary,
        "median_distance_ratio_prediction_over_sample": (
            prediction_summary["median_m"] / train_summary["median_m"]
        ),
        "p90_distance_ratio_prediction_over_sample": (
            prediction_summary["p90_m"] / train_summary["p90_m"]
        ),
        "limitations": [
            "This diagnoses the current frozen split, not deployment across all of Berlin.",
            "Tile-centre distance is a spatial proxy and does not measure spectral or temporal shift.",
            "The 2025 orthophotos are not included because their deployment grid is not prepared here.",
        ],
    }
    (output_dir / "spatial_distance_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_source_data(
        output_dir / "spatial_distance_source.csv",
        train_tiles,
        prediction_tiles,
        train_distances,
        prediction_distances,
    )
    make_figure(
        output_dir / "spatial_distance_distribution",
        train_points,
        prediction_points,
        train_distances,
        prediction_distances,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
