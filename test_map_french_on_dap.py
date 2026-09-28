import unittest
from pathlib import Path

import numpy as np

from map_french_on_dap import (
    DAP_EXPERIMENT,
    choose_knn_size,
    load_coordinates,
    load_distance_matrix,
    orient_coordinates,
    pool_and_project,
    select_landmarks,
)
from sample_approval_rankings import parse_preflib_cat


class FrenchDapMapTests(unittest.TestCase):
    def test_pooling_projects_to_eight_most_approved_candidates(self) -> None:
        paths = sorted(Path("00026_frenchapproval").glob("*.cat"))
        elections = [parse_preflib_cat(path) for path in paths]
        projected = pool_and_project(elections, "French_test", "French test")

        self.assertEqual(projected.election.num_candidates, 8)
        self.assertEqual(
            projected.election.num_voters,
            sum(election.num_voters for election in elections),
        )
        self.assertEqual(len(projected.original_candidate_ids), 8)
        self.assertTrue(
            set(candidate for ballot in projected.election.ballots for candidate in ballot.approved)
            <= set(range(8))
        )

    def test_dap_landmarks_and_knn_calibration(self) -> None:
        instance_ids, coordinates = load_coordinates(
            DAP_EXPERIMENT / "coordinates" / "mds_swap_2d.csv"
        )
        distances = load_distance_matrix(
            DAP_EXPERIMENT / "distances" / "swap.csv", instance_ids
        )
        landmarks = select_landmarks(instance_ids, coordinates, 10)
        k, errors = choose_knn_size(distances[:, landmarks], coordinates)

        self.assertEqual(len(instance_ids), 302)
        self.assertEqual(distances.shape, (302, 302))
        self.assertEqual(len(set(landmarks)), 10)
        self.assertIn(instance_ids.index("ID"), landmarks)
        self.assertIn(k, errors)
        self.assertTrue(all(np.isfinite(error) for error in errors.values()))

    def test_orientation_preserves_pairwise_distances(self) -> None:
        points = np.asarray(((1.0, 2.0), (4.0, 6.0), (-2.0, 5.0)))
        oriented = orient_coordinates(points, points[0])
        before = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        after = np.linalg.norm(oriented[:, None] - oriented[None, :], axis=2)
        np.testing.assert_allclose(before, after)


if __name__ == "__main__":
    unittest.main()
