import unittest

import numpy as np

from candidate_pf_pareto import candidate_curves, dominance, quota_intervals
from experiment_pf_by_level import exact_pf_rho
from proportional_audit import audit_exact


class CandidateParetoTests(unittest.TestCase):
    def test_weighted_curves_match_expanded_electorate(self):
        voters = np.array([[0, 0], [0, 0], [0, 1], [1, 0], [1, 1], [1, 1]])
        for metric in ("hamming", "euclidean"):
            curves = candidate_curves(voters, metric)
            for candidate in range(voters.shape[1]):
                chosen = voters[voters[:, candidate] == 1]
                potential = voters.copy()
                potential[:, candidate] = 1
                selected_distances = np.count_nonzero(
                    voters[:, None, :] != chosen[None, :, :], axis=2).min(axis=1)
                deviation_distances = np.count_nonzero(
                    voters[:, None, :] != potential[None, :, :], axis=2)
                if metric == "euclidean":
                    selected_distances = np.sqrt(selected_distances)
                    deviation_distances = np.sqrt(deviation_distances)
                for index, (_, _, quota) in enumerate(quota_intervals(len(voters))):
                    self.assertEqual(curves[candidate, index], exact_pf_rho(
                        selected_distances, deviation_distances, quota))
                if metric == "euclidean":
                    k = len(chosen)
                    index = next(i for i, (first, last, _) in enumerate(
                        quota_intervals(len(voters))) if first <= k <= last)
                    self.assertEqual(curves[candidate, index], audit_exact(
                        voters, chosen, potential).rho)

    def test_empty_and_unanimous_candidates(self):
        curves = candidate_curves(np.array([[0, 1], [0, 1], [0, 1]]))
        self.assertTrue(np.all(np.isinf(curves[0])))
        np.testing.assert_array_equal(curves[1], 1)

    def test_nonbinary_matrix_rejected(self):
        with self.assertRaisesRegex(ValueError, "binary"):
            candidate_curves(np.array([[0, 2]]))

    def test_intervals_cover_every_k_with_exact_quota(self):
        for n in range(1, 101):
            expanded = [(k, quota) for first, last, quota in quota_intervals(n)
                        for k in range(first, last + 1)]
            self.assertEqual(expanded, [(k, (n + k - 1) // k)
                                        for k in range(1, n + 1)])
            quotas = [q for _, _, q in quota_intervals(n)]
            self.assertEqual(len(quotas), len(set(quotas)))

    def test_dominance_requires_same_challenger_at_every_k(self):
        weak, strict = dominance(np.array([[2, 2], [1, 3], [3, 1], [2, 2], [1, 2]]))
        self.assertEqual(np.flatnonzero(weak[:, 0]).tolist(), [3, 4])
        self.assertEqual(np.flatnonzero(strict[:, 0]).tolist(), [4])
        self.assertFalse(np.any(np.diag(weak)))

    def test_infinity_ties_and_improvements(self):
        weak, strict = dominance(np.array([[1, np.inf], [1, np.inf], [1, 2]]))
        self.assertTrue(weak[0, 1])
        self.assertFalse(strict[0, 1])
        self.assertTrue(strict[2, 0])


if __name__ == "__main__":
    unittest.main()
