from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
from PIL import Image
from rasterio.transform import from_bounds
from shapely.geometry import Polygon, box


from green_roofs.labels import read_polygon_shapefile
from green_roofs.rasterization import rasterize_geometries
from green_roofs.spatial_split import assign_spatial_splits
from green_roofs.study_area import select_tiles_by_centre
from green_roofs.tiles import TileKey, tile_bounds_mercator
from green_roofs.label_qa import (
    ReviewRecord,
    coverage_stratified_sample,
    create_review_pair,
)
from green_roofs.qgis_annotation import (
    create_annotation_geopackage,
    mosaic_spec,
)


class CoordinateTests(unittest.TestCase):
    def test_mercantile_tile_bounds_have_positive_extent(self) -> None:
        left, bottom, right, top = tile_bounds_mercator(TileKey(281790, 172160))
        self.assertLess(left, right)
        self.assertLess(bottom, top)
        self.assertAlmostEqual(right - left, top - bottom, places=8)

    def test_qgis_mosaic_spec_matches_rectangular_xyz_grid(self) -> None:
        keys = {
            TileKey(x, y)
            for x in range(281761, 281815)
            for y in range(172126, 172191)
        }
        spec = mosaic_spec(keys)
        self.assertEqual((spec.width, spec.height), (54 * 256, 65 * 256))
        self.assertEqual((spec.min_x, spec.max_x), (281761, 281814))
        self.assertEqual((spec.min_y, spec.max_y), (172126, 172190))

    def test_qgis_mosaic_rejects_missing_tile(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing 1 tiles"):
            mosaic_spec({TileKey(10, 20), TileKey(11, 20), TileKey(10, 21)})

    def test_study_area_selects_tiles_by_centre(self) -> None:
        inside = TileKey(281790, 172160)
        outside = TileKey(281791, 172160)
        selected = select_tiles_by_centre(
            {inside: Path("inside.png"), outside: Path("outside.png")},
            box(*tile_bounds_mercator(inside)),
        )
        self.assertEqual(selected, {inside: Path("inside.png")})


class QGISAnnotationTests(unittest.TestCase):
    def test_geopandas_writes_label_and_tile_grid_layers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entire_root = root / "training_labels_entire_roofs"
            partial_root = root / "training_labels_partial_roofs"
            entire_root.mkdir()
            partial_root.mkdir()
            geometry = Polygon(
                [(13.48, 52.42), (13.481, 52.42), (13.481, 52.421), (13.48, 52.42)]
            )
            gpd.GeoDataFrame(
                {"plz": ["12357"], "bds16_id": [1]},
                geometry=[geometry],
                crs="EPSG:4326",
            ).to_file(entire_root / "Berlin_2016_12357_Green.shp", engine="pyogrio")
            gpd.GeoDataFrame(
                {"plz": ["12357"], "gt2016_id": [2], "gruen_kat": ["extensiv"]},
                geometry=[geometry],
                crs="EPSG:4326",
            ).to_file(
                partial_root / "Berlin_Gründächer_DachteilflächenGebäude_2016_12357.shp",
                engine="pyogrio",
            )
            output_path = root / "annotations.gpkg"

            counts = create_annotation_geopackage(
                root,
                {TileKey(281790, 172160): root / "unused.png"},
                output_path,
            )

            labels = gpd.read_file(output_path, layer="green_roofs", engine="pyogrio")
            tiles = gpd.read_file(output_path, layer="tile_grid", engine="pyogrio")
            self.assertEqual(counts, {"label_feature_count": 2, "tile_feature_count": 1})
            self.assertEqual(len(labels), 2)
            self.assertEqual(len(tiles), 1)
            self.assertEqual(labels.crs.to_epsg(), 3857)
            self.assertEqual(tiles.crs.to_epsg(), 3857)


class RasterizationTests(unittest.TestCase):
    def test_geopandas_reader_preserves_polygon_and_hole_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "polygons.shp"
            polygon = Polygon(
                [(0, 0), (0, 4), (4, 4), (4, 0), (0, 0)],
                [[(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)]],
            )
            source = gpd.GeoDataFrame(
                {"name": ["test"]}, geometry=[polygon], crs="EPSG:4326"
            )
            source.to_file(path, engine="pyogrio")

            features = read_polygon_shapefile(path)

            self.assertEqual(len(features), 1)
            self.assertEqual(features.geometry.iloc[0].geom_type, "Polygon")
            self.assertEqual(len(features.geometry.iloc[0].interiors), 1)
            self.assertEqual(features.crs.to_epsg(), 3857)

    def test_rasterio_preserves_polygon_hole(self) -> None:
        key = TileKey(281790, 172160)
        transform = from_bounds(*tile_bounds_mercator(key), 256, 256)
        exterior = [
            transform * point
            for point in [(20, 20), (220, 20), (220, 220), (20, 220)]
        ]
        hole = [
            transform * point
            for point in [(80, 80), (80, 180), (180, 180), (180, 80)]
        ]
        geometry = Polygon(exterior, [hole])
        mask = rasterize_geometries([geometry], key)

        self.assertEqual(mask.getpixel((40, 40)), 1)
        self.assertEqual(mask.getpixel((128, 128)), 0)
        self.assertGreater(mask.histogram()[1], 0)
        self.assertEqual(mask.getextrema(), (0, 1))


class SpatialSplitTests(unittest.TestCase):
    def test_tiles_in_one_block_share_a_split(self) -> None:
        tile_stats = {
            TileKey(x, y): (x + y) % 17
            for x in range(100, 124)
            for y in range(200, 224)
        }
        assignments, _ = assign_spatial_splits(tile_stats, block_size=8, seed=42)
        block_splits: dict[str, set[str]] = {}
        for split, block_id in assignments.values():
            block_splits.setdefault(block_id, set()).add(split)
        self.assertTrue(all(len(splits) == 1 for splits in block_splits.values()))
        self.assertEqual({split for split, _ in assignments.values()}, {"train", "val", "test"})


class LabelQualityReviewTests(unittest.TestCase):
    def test_coverage_stratified_sample_is_unique_and_reproducible(self) -> None:
        records = [
            ReviewRecord(
                tile_id=f"tile_{index:03d}",
                split="train",
                image_relpath=f"images/{index}.png",
                mask_relpath=f"masks/{index}.png",
                positive_pixels=index + 1,
                positive_fraction=(index + 1) / 100,
            )
            for index in range(100)
        ]
        first = coverage_stratified_sample(records, sample_count=20, seed=42)
        second = coverage_stratified_sample(records, sample_count=20, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        self.assertEqual(len({record.tile_id for record in first}), 20)
        self.assertLess(min(record.positive_fraction for record in first), 0.06)
        self.assertGreater(max(record.positive_fraction for record in first), 0.95)

    def test_review_pair_has_two_panels_and_preserves_inputs(self) -> None:
        record = ReviewRecord(
            tile_id="19_1_2",
            split="train",
            image_relpath="image.png",
            mask_relpath="mask.png",
            positive_pixels=100,
            positive_fraction=100 / (256 * 256),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.png"
            mask_path = root / "mask.png"
            Image.new("RGB", (256, 256), (80, 90, 100)).save(image_path)
            mask = Image.new("L", (256, 256), 0)
            for x in range(10, 20):
                for y in range(10, 20):
                    mask.putpixel((x, y), 1)
            mask.save(mask_path)

            pair = create_review_pair(image_path, mask_path, record)

            self.assertEqual(pair.size, (512, 308))
            self.assertEqual(Image.open(image_path).getpixel((0, 0)), (80, 90, 100))
            self.assertEqual(Image.open(mask_path).getpixel((10, 10)), 1)


if __name__ == "__main__":
    unittest.main()
