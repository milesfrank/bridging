import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from clear_communities import (
    find_clear_communities,
    find_partition,
    write_communities,
    write_partition,
)


class ClearCommunityTests(unittest.TestCase):
    def test_finds_nested_clear_communities_and_expands_duplicate_rows(self) -> None:
        voters = np.asarray(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
                [1, 1, 1],
            ]
        )

        communities = find_clear_communities(voters)

        self.assertEqual(
            [(item.voter_rows, item.diameter, item.min_external_distance)
             for item in communities],
            [((1, 2), 0, 1), ((1, 2, 3), 1, 2)],
        )

    def test_rejects_connected_component_that_is_not_a_clique(self) -> None:
        voters = np.asarray([[0, 0], [0, 1], [1, 1], [1, 0]])
        self.assertEqual(find_clear_communities(voters), [])

    def test_whole_population_is_not_a_community(self) -> None:
        voters = np.asarray([[0], [0], [0]])
        self.assertEqual(find_clear_communities(voters), [])

    def test_compact_output_records_every_qualifying_q_as_a_range(self) -> None:
        community = find_clear_communities(
            np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 1], [1, 1, 1]])
        )[1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "communities.csv"
            write_communities([community], path)
            with path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
        self.assertEqual((row["q_min"], row["q_max"]), ("2", "3"))
        self.assertEqual(row["voter_rows"], "1;2;3")

    def test_maximal_clear_communities_form_partition(self) -> None:
        voters = np.asarray(
            [[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]]
        )
        partition = find_partition(find_clear_communities(voters), len(voters))
        self.assertIsNotNone(partition)
        self.assertEqual(
            {item.voter_rows for item in partition or []},
            {(1, 2), (3, 4)},
        )

    def test_reports_when_no_clear_community_partition_exists(self) -> None:
        voters = np.asarray([[0, 0], [0, 1], [1, 1], [1, 0]])
        self.assertIsNone(
            find_partition(find_clear_communities(voters), len(voters))
        )

    def test_partition_output_records_common_q_range(self) -> None:
        voters = np.asarray(
            [[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
        )
        partition = find_partition(find_clear_communities(voters), len(voters))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partition.csv"
            write_partition(partition, path)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertTrue(
            all(
                (row["partition_q_min"], row["partition_q_max"])
                == ("2", "2")
                for row in rows
            )
        )


if __name__ == "__main__":
    unittest.main()
