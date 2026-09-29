import itertools
import unittest
from fractions import Fraction

import numpy as np

from ballot_distances import row_distances
from generic_dc_mpjr_min_gamma import minimum_gamma_dc_mpjr_indices
from dc_mpjr_verify import verify_gamma_dc_mpjr
from clear_communities import find_clear_communities
from candidate_pf_pareto import candidate_curves, quota_intervals
from experiment_pf_by_level import exact_pf_rho
from representation_level_audit import representation_level_audit
from wasserstein_approver_distance import wasserstein_to_approvers
from plot_voter_dendrogram import metrics_by_height
from nested_greedy_capture import construct_ballot_levels, construct_levels


class JaccardOptionsTests(unittest.TestCase):
    def test_compressed_capture_preserves_ties_and_witnesses(self):
        rng = np.random.default_rng(42)
        for metric in ('hamming', 'jaccard', 'euclidean'):
            for _ in range(5):
                voters = rng.integers(0, 2, (18, 3))
                self.assertEqual(construct_ballot_levels(voters, metric),
                                 construct_levels(row_distances(voters, voters, metric)))

    def test_empty_and_fractional_distances(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        np.testing.assert_array_equal(row_distances(points, points, 'jaccard'),
                                      [[0, 1, 1], [1, 0, .5], [1, .5, 0]])
        with self.assertRaises(ValueError):
            row_distances(np.array([[2]]), np.array([[0]]), 'jaccard')

    def test_gamma_against_expanded_ball_enumeration(self):
        rng = np.random.default_rng(10)
        for metric in ('hamming', 'jaccard'):
            for _ in range(30):
                agents = rng.integers(0, 2, (8, 4))
                candidates = rng.integers(0, 2, (7, 4))
                selected = [0, 2, 4]
                d = row_distances(agents, candidates, metric)
                expected = Fraction(1)
                for c in [1, 3, 5, 6]:
                    for radius in np.unique(d[:, c]):
                        members = d[:, c] <= radius
                        deserved = int(members.sum()) * len(selected) // len(agents)
                        if not deserved:
                            continue
                        required = np.sort(d[members][:, selected].min(axis=0))[deserved - 1]
                        if radius == 0:
                            if required > 0:
                                expected = None
                                break
                        else:
                            ratio = Fraction(float(required)).limit_denominator(4) / Fraction(float(radius)).limit_denominator(4)
                            expected = max(expected, ratio)
                    if expected is None:
                        break
                result = minimum_gamma_dc_mpjr_indices(agents, candidates, selected, metric)
                self.assertEqual(result.gamma, expected)
                self.assertEqual(verify_gamma_dc_mpjr(agents, candidates, selected, 1, metric).satisfies,
                                 expected == 1)
                if expected is not None:
                    self.assertTrue(verify_gamma_dc_mpjr(agents, candidates, selected, float(expected), metric).satisfies)

    def test_communities_against_subsets(self):
        voters = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1]])
        d = row_distances(voters, voters, 'jaccard')
        expected = set()
        for size in range(2, len(voters)):
            for members in itertools.combinations(range(len(voters)), size):
                outside = sorted(set(range(len(voters))) - set(members))
                if d[np.ix_(members, members)].max() < d[np.ix_(members, outside)].min():
                    expected.add(tuple(i + 1 for i in members))
        actual = find_clear_communities(voters, 'jaccard')
        self.assertEqual({c.voter_rows for c in actual}, expected)
        self.assertTrue(any(c.diameter == .5 for c in actual))

    def test_pf_curves_and_representation(self):
        voters = np.array([[0, 0], [0, 0], [1, 0], [1, 1], [0, 1]])
        curves = candidate_curves(voters, 'jaccard')
        for c in range(2):
            chosen = voters[voters[:, c] == 1]
            potential = voters.copy(); potential[:, c] = 1
            chosen_d = row_distances(voters, chosen, 'jaccard')
            candidate_d = row_distances(voters, potential, 'jaccard')
            for i, (_, _, q) in enumerate(quota_intervals(len(voters))):
                self.assertEqual(curves[c, i], exact_pf_rho(chosen_d.min(axis=1), candidate_d, q))
            result = representation_level_audit(voters, chosen, metric='jaccard')[0]
            radii = np.sort(row_distances(voters, voters, 'jaccard'), axis=1)[:, 2]
            self.assertEqual(result.satisfied_voter_count, np.count_nonzero(chosen_d.min(axis=1) <= radii))

    def test_transport_and_fractional_heights(self):
        voters = np.array([[1, 0], [1, 1]])
        self.assertEqual(wasserstein_to_approvers(voters, 1, metric='jaccard').distance, .25)
        linkage = np.array([[0, 1, .25, 2], [2, 3, .75, 3]])
        rows = metrics_by_height(linkage, 3, .5, 'jaccard')
        self.assertEqual([r[0] for r in rows], [0, .25, .5, .75])

    def test_weighted_pf_matches_expansion(self):
        rng = np.random.default_rng(91)
        points = rng.integers(0, 2, (9, 5))
        counts = rng.integers(1, 5, 9)
        distances = row_distances(points, points, 'jaccard')
        nearest = distances[:, [1, 4]].min(axis=1)
        for quota in range(1, int(counts.sum()) + 1):
            self.assertEqual(exact_pf_rho(nearest, distances, quota, counts),
                             exact_pf_rho(np.repeat(nearest, counts),
                                          np.repeat(distances, counts, axis=0), quota))


if __name__ == '__main__':
    unittest.main()
