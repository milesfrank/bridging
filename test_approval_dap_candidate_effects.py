import unittest
from pathlib import Path

import numpy as np

from approval_dap_candidate_effects import (
    approval_distance_matrix,
    calculate_scores,
    greedy_k_medoids_costs,
    remove_candidate,
)
from sample_approval_rankings import ApprovalBallot, ApprovalElection


class ApprovalDapCandidateEffectsTests(unittest.TestCase):
    def _election(self, ballots, candidates=("A", "B")) -> ApprovalElection:
        approval_ballots = tuple(
            ApprovalBallot(count, tuple(approved)) for count, approved in ballots
        )
        return ApprovalElection(
            path=Path("test.cat"),
            title="Test",
            candidate_names=tuple(candidates),
            ballots=approval_ballots,
            num_voters=sum(count for count, _ in ballots),
        )

    def test_identity_has_full_agreement_and_no_diversity(self) -> None:
        election = self._election(((5, (0,)),))
        for metric in ("hamming", "jaccard"):
            scores = calculate_scores(election, metric)
            self.assertAlmostEqual(scores.agreement, 1.0)
            self.assertAlmostEqual(scores.diversity, 0.0)
            self.assertAlmostEqual(scores.polarization, 0.0)

    def test_two_opposing_groups_match_paper_anchors(self) -> None:
        election = self._election(((2, (0,)), (2, (1,))))
        for metric in ("hamming", "jaccard"):
            scores = calculate_scores(election, metric)
            self.assertAlmostEqual(scores.agreement, 0.0)
            self.assertAlmostEqual(scores.diversity, 0.5)
            self.assertAlmostEqual(scores.polarization, 1.0)

    def test_candidate_removal_merges_ballot_types_and_preserves_voters(self) -> None:
        election = self._election(
            ((2, (0,)), (3, (0, 1)), (4, (1,))),
            candidates=("A", "B", "C"),
        )
        projected = remove_candidate(election, 0)
        self.assertEqual(projected.num_voters, 9)
        self.assertEqual(projected.candidate_names, ("B", "C"))
        self.assertEqual(
            {(ballot.approved, ballot.count) for ballot in projected.ballots},
            {((), 2), ((0,), 7)},
        )

    def test_weighted_greedy_cost_curve(self) -> None:
        types = np.asarray(((1, 0), (0, 1)))
        weights = np.asarray((2.0, 2.0))
        distances = approval_distance_matrix(types, "hamming")
        costs, chosen = greedy_k_medoids_costs(distances, weights)
        np.testing.assert_allclose(costs, (4.0, 0.0))
        self.assertEqual(set(chosen), {0, 1})


if __name__ == "__main__":
    unittest.main()
