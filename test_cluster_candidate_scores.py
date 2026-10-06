import unittest

import numpy as np

from cluster_candidate_scores import (
    candidate_distances, cluster_voters, correlation_objective,
    score_candidates, voter_distances,
)


class ClusterCandidateScoresTests(unittest.TestCase):
    def test_direct_voter_distances(self):
        matrix = [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]]
        np.testing.assert_allclose(voter_distances(matrix, "hamming"),
                                   [[0, .25, .5, 0], [.25, 0, .25, .25],
                                    [.5, .25, 0, .5], [0, .25, .5, 0]])
        np.testing.assert_allclose(voter_distances(matrix, "jaccard"),
                                   [[0, 1, 1, 0], [1, 0, .5, 1],
                                    [1, .5, 0, 1], [0, 1, 1, 0]])
        # The same cutoff can produce different partitions under the metrics.
        self.assertEqual(len(set(cluster_voters(voter_distances(matrix, "hamming"), cutoff=.5))), 1)
        self.assertEqual(len(set(cluster_voters(voter_distances(matrix, "jaccard"), cutoff=.5))), 2)
        for distance in ("disagreement", "hamming", "jaccard"):
            np.testing.assert_array_equal(voter_distances([[0, 0]], distance), [[0]])
            with self.assertRaises(ValueError):
                voter_distances([[.5, 0]], distance)
        with self.assertRaises(ValueError):
            voter_distances(matrix, "unknown")

    def test_candidate_sets_and_single_disagreement(self):
        matrix = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]])
        j = candidate_distances(matrix)
        self.assertEqual(j[0, 1], 0.5)
        self.assertEqual(j[2, 3], 0)
        np.testing.assert_array_equal(voter_distances(matrix),
                                      [[0, 0, 0.5], [0, 0, 0], [0.5, 0, 0]])

    def test_distances_against_set_definition(self):
        matrix = np.random.default_rng(8).integers(0, 2, (30, 5))
        approvers = [set(np.flatnonzero(col)) for col in matrix.T]
        expected = np.zeros((len(matrix), len(matrix)))
        for x in range(len(matrix)):
            for y in range(len(matrix)):
                disagree = np.flatnonzero(matrix[x] != matrix[y])
                for d in disagree:
                    for e in disagree:
                        union = approvers[d] | approvers[e]
                        distance = len(approvers[d] ^ approvers[e]) / len(union) if union else 0
                        expected[x, y] = max(expected[x, y], distance)
        np.testing.assert_allclose(voter_distances(matrix), expected)

    def test_linkage_cutoffs(self):
        distances = np.array([[0, .2, .6], [.2, 0, .8], [.6, .8, 0]])
        self.assertEqual(len(set(cluster_voters(distances, "complete", .75))), 2)
        self.assertEqual(len(set(cluster_voters(distances, "average", .75))), 1)
        self.assertEqual(len(set(cluster_voters(distances, "complete", .2))), 2)

    def test_correlation_merge_objective(self):
        distances = np.array([[0, .1, .9], [.1, 0, .8], [.9, .8, 0]])
        labels = cluster_voters(distances, "correlation")
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])
        self.assertAlmostEqual(correlation_objective(distances, labels), .4)

    def test_welfare_with_unequal_groups_and_zero_approval(self):
        matrix = np.array([[1, 1], [0, 1], [1, 0]])
        labels = np.array([7, 7, 9])
        np.testing.assert_allclose(score_candidates(matrix, labels, "pairwise"), [1, 0])
        np.testing.assert_allclose(score_candidates(matrix, labels, "nash"), [.5 ** (2/3), 0])
        np.testing.assert_allclose(score_candidates(matrix, labels, "egalitarian"), [.5, 0])

    def test_one_voter_and_one_group(self):
        distances = voter_distances([[1, 0]])
        for method in ("complete", "average", "correlation"):
            np.testing.assert_array_equal(cluster_voters(distances, method), [0])
        np.testing.assert_array_equal(score_candidates([[1, 0]], [0]), [0, 0])
        np.testing.assert_allclose(score_candidates([[1], [0]], [0, 0], "nash"), [.5])

    def test_reject_nonbinary_before_integer_conversion(self):
        for matrix in ([[256]], [[.5]], [[float("nan")]], []):
            with self.assertRaises(ValueError):
                voter_distances(matrix)


if __name__ == "__main__":
    unittest.main()
