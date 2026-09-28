import random
import tempfile
import unittest
from pathlib import Path

from sample_approval_rankings import (
    parse_preflib_cat,
    sample_rankings,
    uniform_completion,
    write_soc,
)


CAT_TEXT = """# TITLE: Tiny approval election
# NUMBER ALTERNATIVES: 4
# NUMBER VOTERS: 3
# NUMBER UNIQUE PREFERENCES: 2
# NUMBER CATEGORIES: 2
# CATEGORY NAME 1: Yes
# CATEGORY NAME 2: No
# ALTERNATIVE NAME 1: A
# ALTERNATIVE NAME 2: B
# ALTERNATIVE NAME 3: C
# ALTERNATIVE NAME 4: D
2: {1,3},{2,4}
1: {},{1,2,3,4}
"""


class ApprovalRankingTests(unittest.TestCase):
    def _write_election(self, directory: str) -> tuple[Path, object]:
        source = Path(directory) / "tiny.cat"
        source.write_text(CAT_TEXT, encoding="utf-8")
        return source, parse_preflib_cat(source)

    def test_parser_and_sampling_are_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, election = self._write_election(directory)

            self.assertEqual(election.title, "Tiny approval election")
            self.assertEqual(election.candidate_names, ("A", "B", "C", "D"))
            self.assertEqual(election.num_voters, 3)
            self.assertEqual(election.ballots[0].approved, (0, 2))

            first = sample_rankings(election, num_rankings=30, seed=7)
            second = sample_rankings(election, num_rankings=30, seed=7)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 30)
            self.assertTrue(
                all(sorted(ranking) == [0, 1, 2, 3] for ranking in first)
            )

    def test_uniform_completion_respects_approval_cut(self) -> None:
        rng = random.Random(11)
        approved = {0, 2}
        for _ in range(100):
            ranking = uniform_completion(tuple(approved), 4, rng)
            positions = {
                candidate: position for position, candidate in enumerate(ranking)
            }
            self.assertLess(
                max(positions[candidate] for candidate in approved),
                min(positions[candidate] for candidate in {1, 3}),
            )

    def test_complete_mode_and_soc_round_trip(self) -> None:
        from map_ordinal_preferences import read_soc

        with tempfile.TemporaryDirectory() as directory:
            _, election = self._write_election(directory)
            rankings = sample_rankings(election, seed=3, mode="complete")
            output = Path(directory) / "tiny.soc"
            write_soc(output, election, rankings, seed=3, mode="complete")

            profile = read_soc(output)
            self.assertEqual(profile.title, election.title)
            self.assertEqual(profile.candidate_names, election.candidate_names)
            self.assertEqual(profile.num_rankings, election.num_voters)

    def test_complete_mode_rejects_different_sample_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, election = self._write_election(directory)
            with self.assertRaisesRegex(ValueError, "exactly one ranking"):
                sample_rankings(election, num_rankings=2, mode="complete")


if __name__ == "__main__":
    unittest.main()
