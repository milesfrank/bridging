import itertools
import unittest
from fractions import Fraction

import numpy as np

from candidate_pairwise_scores import candidate_pairwise_scores


class CandidatePairwiseScoresTests(unittest.TestCase):
    def test_bridge_between_unequal_opposing_sets(self):
        # A={0,1}, B={2}, C={0,2}. Each candidate receives a contribution
        # only from the comparison of the other two approval sets.
        matrix = np.array([[1, 0, 1], [1, 0, 0], [0, 1, 1]])
        np.testing.assert_allclose(candidate_pairwise_scores(matrix), [0, 0, 1/9])

    def test_duplicate_candidates_do_not_multiply_comparisons(self):
        matrix = np.array([[1, 0, 1], [1, 0, 0], [0, 1, 1]])
        scores = candidate_pairwise_scores(matrix)
        cloned = candidate_pairwise_scores(matrix[:, [0, 1, 2, 0, 2]])
        np.testing.assert_allclose(cloned, scores[[0, 1, 2, 0, 2]])

    def test_replication_invariance(self):
        matrix = np.random.default_rng(11).integers(0, 2, (13, 6))
        np.testing.assert_allclose(candidate_pairwise_scores(np.repeat(matrix, 5, axis=0)),
                                   candidate_pairwise_scores(matrix))

    def test_against_literal_set_formula(self):
        rng = np.random.default_rng(4)
        for _ in range(30):
            matrix = rng.integers(0, 2, (8, 7))
            approvals = [frozenset(np.flatnonzero(col)) for col in matrix.T]
            expected = [Fraction(0) for _ in approvals]
            for s, t in itertools.combinations(set(approvals), 2):
                u, v = s - t, t - s
                for c, a in enumerate(approvals):
                    expected[c] += Fraction(min(len(v) * len(a & u), len(u) * len(a & v)), len(matrix)**2)
            np.testing.assert_allclose(candidate_pairwise_scores(matrix), list(map(float, expected)))

    def test_nested_empty_universal_and_single_sets(self):
        for matrix in ([[0, 1, 1], [0, 0, 1]], [[1, 0]], [[0], [1]], [[1, 1], [0, 0]]):
            np.testing.assert_array_equal(candidate_pairwise_scores(matrix), np.zeros(len(matrix[0])))

    def test_invalid_input(self):
        for matrix in ([], [[]], [[.5]], [[256]], [[float("nan")]]):
            with self.assertRaises(ValueError):
                candidate_pairwise_scores(matrix)


if __name__ == "__main__":
    unittest.main()
