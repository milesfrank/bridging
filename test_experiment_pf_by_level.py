import unittest

import numpy as np

from experiment_pf_by_level import exact_pf_rho, run_trials


class PfExperimentTests(unittest.TestCase):
    def test_all_voters_selected_is_one_pf(self) -> None:
        distances = np.asarray([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        self.assertEqual(exact_pf_rho(np.zeros(3), distances, quota=1), 1.0)

    def test_explicit_quota_is_used(self) -> None:
        distances = np.asarray([[0, 2], [1, 1], [2, 0]])
        selected = np.asarray([0, 1, 2])
        self.assertEqual(exact_pf_rho(selected, distances, quota=2), 1.0)
        self.assertTrue(np.isinf(exact_pf_rho(selected, distances, quota=1)))

    def test_trials_are_reproducible(self) -> None:
        distances = np.asarray([[0, 1], [1, 0]])
        groups = ((0, 1),)
        voter_type = np.asarray([0, 1])
        first = run_trials(groups, 2, voter_type, distances, 10, np.random.default_rng(7))
        second = run_trials(groups, 2, voter_type, distances, 10, np.random.default_rng(7))
        np.testing.assert_array_equal(first.rhos, second.rhos)
        np.testing.assert_array_equal(
            first.distinct_representatives, second.distinct_representatives
        )


if __name__ == "__main__":
    unittest.main()
