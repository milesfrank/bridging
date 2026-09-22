import math
import unittest

import numpy as np

from nested_greedy_capture import (
    augmented_greedy_capture,
    construct_levels,
    pairwise_distances,
    validate_levels,
)


class NestedGreedyCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.points = np.asarray([[0.0], [1.0], [3.0], [7.0], [8.0]])
        self.distances = pairwise_distances(self.points, "euclidean")

    def test_capture_is_disjoint_and_quota_sized(self) -> None:
        captures = augmented_greedy_capture(self.distances, 2)
        self.assertEqual(len(captures), 2)
        members = [set(capture.members) for capture in captures]
        self.assertTrue(all(len(group) == 2 for group in members))
        self.assertTrue(members[0].isdisjoint(members[1]))

    def test_every_level_satisfies_cardinality_invariants(self) -> None:
        levels = construct_levels(self.distances)
        validate_levels(levels, len(self.points))
        for level in levels:
            self.assertLessEqual(len(level.groups), level.k)
            self.assertTrue(
                all(len(group) == math.ceil(len(self.points) / level.k) for group in level.groups)
            )

    def test_equal_quota_copies_groups_and_alpha(self) -> None:
        levels = {level.k: level for level in construct_levels(self.distances)}
        self.assertEqual(levels[3].quota, levels[4].quota)
        self.assertTrue(levels[3].copied_from_next_level)
        self.assertIs(levels[3].groups, levels[4].groups)
        self.assertEqual(levels[3].alpha, levels[4].alpha)

    def test_hamming_counts_unequal_coordinates(self) -> None:
        points = np.asarray([[0, 1, 1], [1, 1, 0]])
        distances = pairwise_distances(points, "hamming")
        self.assertEqual(distances[0, 1], 2)


if __name__ == "__main__":
    unittest.main()
