from __future__ import annotations

import unittest

import numpy as np


from green_roofs.spatial_distance import (
    log_distance_kde,
    nearest_distances,
    tile_centres_to_projected,
)


class SpatialDistanceTests(unittest.TestCase):
    def test_nearest_distances_exclude_self(self) -> None:
        points = np.array([[0.0, 0.0], [3.0, 0.0], [10.0, 0.0]])
        distances = nearest_distances(points, points, exclude_matching_index=True)
        np.testing.assert_allclose(distances, [3.0, 3.0, 7.0])

    def test_query_to_reference_distances(self) -> None:
        reference = np.array([[0.0, 0.0], [10.0, 0.0]])
        query = np.array([[2.0, 0.0], [9.0, 0.0]])
        np.testing.assert_allclose(nearest_distances(query, reference), [2.0, 1.0])

    def test_adjacent_berlin_tiles_have_plausible_ground_spacing(self) -> None:
        tile_xy = np.array([[281780, 172150], [281781, 172150]])
        projected = tile_centres_to_projected(tile_xy)
        distance = float(np.linalg.norm(projected[1] - projected[0]))
        self.assertGreater(distance, 40.0)
        self.assertLess(distance, 55.0)

    def test_log_kde_is_finite_for_repeated_grid_distances(self) -> None:
        distances = np.full(20, 46.6)
        grid = np.linspace(1.0, 3.0, 100)
        density = log_distance_kde(distances, grid)
        self.assertTrue(np.all(np.isfinite(density)))
        self.assertGreater(float(np.max(density)), 0.0)
