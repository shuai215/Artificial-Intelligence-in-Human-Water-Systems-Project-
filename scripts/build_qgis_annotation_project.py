"""Build the ready-to-open QGIS project using PyQGIS."""

from __future__ import annotations

import argparse
from pathlib import Path

from qgis.core import (
    Qgis,
    QgsCategorizedSymbolRenderer,
    QgsCoordinateReferenceSystem,
    QgsFillSymbol,
    QgsProject,
    QgsRasterLayer,
    QgsRendererCategory,
    QgsVectorLayer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("dataset_root", type=Path)
    return parser.parse_args()


def require_valid(layer: object, name: str) -> None:
    if not layer.isValid():
        raise RuntimeError(f"QGIS could not load {name}")


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    project = QgsProject.instance()
    project.clear()
    project.setCrs(QgsCoordinateReferenceSystem("EPSG:3857"))
    project.setFilePathStorage(Qgis.FilePathType.Relative)

    mosaic = QgsRasterLayer(str(output_root / "orthophotos_2016_mosaic.vrt"), "2016 orthophoto mosaic")
    labels = QgsVectorLayer(
        f"{output_root / 'green_roofs_labels_v2.gpkg'}|layername=green_roofs",
        "green roofs — editable v2",
        "ogr",
    )
    tile_grid = QgsVectorLayer(
        f"{output_root / 'green_roofs_labels_v2.gpkg'}|layername=tile_grid",
        "XYZ tile grid — reference",
        "ogr",
    )
    postcode_boundary = QgsVectorLayer(
        f"{output_root / 'berlin_plz_12357.gpkg'}|layername=plz_12357",
        "Berlin PLZ 12357 boundary — current reference",
        "ogr",
    )
    buildings = QgsVectorLayer(
        str(args.dataset_root.resolve() / "building_footprints_2016_12357.geojson"),
        "building footprints 2016 — reference",
        "ogr",
    )
    entire_roof_source = QgsVectorLayer(
        str(
            args.dataset_root.resolve()
            / "training_labels_entire_roofs"
            / "Berlin_2016_12357_Green.shp"
        ),
        "source labels — entire roofs (read-only)",
        "ogr",
    )
    partial_roof_source = QgsVectorLayer(
        str(
            args.dataset_root.resolve()
            / "training_labels_partial_roofs"
            / "Berlin_Gründächer_DachteilflächenGebäude_2016_12357.shp"
        ),
        "source labels — partial roofs (read-only)",
        "ogr",
    )
    for layer, name in (
        (mosaic, "mosaic"),
        (labels, "labels"),
        (tile_grid, "tile grid"),
        (postcode_boundary, "postcode boundary"),
        (buildings, "building footprints"),
        (entire_roof_source, "entire-roof source labels"),
        (partial_roof_source, "partial-roof source labels"),
    ):
        require_valid(layer, name)

    labels.setRenderer(
        QgsCategorizedSymbolRenderer(
            "review_status",
            [
                QgsRendererCategory(
                    "unreviewed",
                    QgsFillSymbol.createSimple(
                        {"color": "255,170,0,75", "outline_color": "255,120,0", "outline_width": "0.45"}
                    ),
                    "Unreviewed existing label",
                ),
                QgsRendererCategory(
                    "confirmed",
                    QgsFillSymbol.createSimple(
                        {"color": "0,210,80,75", "outline_color": "0,160,60", "outline_width": "0.45"}
                    ),
                    "Confirmed",
                ),
                QgsRendererCategory(
                    "corrected",
                    QgsFillSymbol.createSimple(
                        {"color": "0,170,255,75", "outline_color": "0,110,220", "outline_width": "0.45"}
                    ),
                    "Corrected or newly added",
                ),
                QgsRendererCategory(
                    "uncertain",
                    QgsFillSymbol.createSimple(
                        {"color": "220,0,220,75", "outline_color": "170,0,170", "outline_width": "0.45"}
                    ),
                    "Uncertain",
                ),
            ],
        )
    )
    tile_grid.setRenderer(
        QgsCategorizedSymbolRenderer(
            "split",
            [
                QgsRendererCategory("train", QgsFillSymbol.createSimple({"color": "0,0,0,0", "outline_color": "60,120,255,80", "outline_width": "0.12"}), "Train"),
                QgsRendererCategory("val", QgsFillSymbol.createSimple({"color": "0,0,0,0", "outline_color": "255,170,0,100", "outline_width": "0.18"}), "Validation"),
                QgsRendererCategory("test", QgsFillSymbol.createSimple({"color": "0,0,0,0", "outline_color": "220,60,60,100", "outline_width": "0.18"}), "Test"),
            ],
        )
    )
    buildings.renderer().setSymbol(
        QgsFillSymbol.createSimple(
            {"color": "0,0,0,0", "outline_color": "255,255,255,100", "outline_width": "0.15"}
        )
    )
    postcode_boundary.renderer().setSymbol(
        QgsFillSymbol.createSimple(
            {
                "color": "0,0,0,0",
                "outline_color": "0,255,255,255",
                "outline_width": "0.8",
            }
        )
    )
    entire_roof_source.setReadOnly(True)
    entire_roof_source.renderer().setSymbol(
        QgsFillSymbol.createSimple(
            {
                "color": "255,230,0,70",
                "outline_color": "255,220,0,255",
                "outline_width": "0.65",
            }
        )
    )
    partial_roof_source.setReadOnly(True)
    partial_roof_source.renderer().setSymbol(
        QgsFillSymbol.createSimple(
            {
                "color": "255,0,180,55",
                "outline_color": "255,0,180,255",
                "outline_width": "0.5",
            }
        )
    )

    for layer in (
        mosaic,
        buildings,
        tile_grid,
        postcode_boundary,
        entire_roof_source,
        partial_roof_source,
        labels,
    ):
        project.addMapLayer(layer)
    project.layerTreeRoot().findLayer(tile_grid.id()).setItemVisibilityChecked(False)
    project.layerTreeRoot().findLayer(buildings.id()).setItemVisibilityChecked(False)
    project.layerTreeRoot().findLayer(entire_roof_source.id()).setItemVisibilityChecked(False)
    project.layerTreeRoot().findLayer(partial_roof_source.id()).setItemVisibilityChecked(False)
    project.write(str(output_root / "green_roofs_reannotation.qgz"))


if __name__ == "__main__":
    main()
