import random
import unittest
from pathlib import Path

import numpy as np

from map_ordinal_preferences import classical_mds
from plot_french_jaccard_mds import (
    jaccard_distances,
    metric_mds,
    normalized_stress,
    sample_approval_ballots,
)
from sample_approval_rankings import ApprovalBallot, ApprovalElection


class FrenchJaccardMdsTests(unittest.TestCase):
    def test_jaccard_distance_and_empty_ballot_convention(self) -> None:
        ballots = np.asarray(
            (
                (0, 0, 0),
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 1),
            )
        )
        distances = jaccard_distances(ballots)
        np.testing.assert_allclose(
            distances,
            (
                (0, 0, 1, 1, 1),
                (0, 0, 1, 1, 1),
                (1, 1, 0, 0.5, 1),
                (1, 1, 0.5, 0, 2 / 3),
                (1, 1, 1, 2 / 3, 0),
            ),
        )

    def test_sampling_is_reproducible_and_without_replacement(self) -> None:
        election = ApprovalElection(
            path=Path("tiny.cat"),
            title="Tiny",
            candidate_names=("A", "B", "C"),
            ballots=(ApprovalBallot(4, (0,)), ApprovalBallot(2, (1, 2))),
            num_voters=6,
        )
        first = sample_approval_ballots(election, 5, random.Random(4))
        second = sample_approval_ballots(election, 5, random.Random(4))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertLessEqual(sum(item.approved == (1, 2) for item in first), 2)

    def test_smacof_does_not_increase_stress(self) -> None:
        ballots = np.asarray(
            (
                (1, 0, 0, 0),
                (1, 1, 0, 0),
                (0, 1, 1, 0),
                (0, 0, 1, 1),
                (1, 0, 0, 1),
                (0, 1, 0, 1),
            )
        )
        distances = jaccard_distances(ballots)
        initial = classical_mds(distances)
        coordinates, iterations = metric_mds(distances, max_iterations=100)
        self.assertEqual(coordinates.shape, (6, 2))
        self.assertGreater(iterations, 0)
        self.assertLessEqual(
            normalized_stress(distances, coordinates),
            normalized_stress(distances, initial) + 1e-12,
        )


if __name__ == "__main__":
    unittest.main()
