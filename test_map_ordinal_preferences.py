import random
import unittest

import numpy as np

from map_ordinal_preferences import (
    classical_mds,
    generate_impartial_culture,
    generate_norm_mallows,
    normalized_kendall_distances,
    phi_from_normalized_dispersion,
)


class OrdinalMapTests(unittest.TestCase):
    def test_kendall_distance_known_rankings(self) -> None:
        rankings = np.asarray(
            [
                [0, 1, 2],
                [0, 2, 1],
                [2, 1, 0],
            ]
        )
        distances = normalized_kendall_distances(rankings)
        np.testing.assert_allclose(
            distances,
            [
                [0.0, 1.0 / 3.0, 1.0],
                [1.0 / 3.0, 0.0, 2.0 / 3.0],
                [1.0, 2.0 / 3.0, 0.0],
            ],
        )

    def test_mallows_dispersion_endpoints(self) -> None:
        self.assertEqual(phi_from_normalized_dispersion(5, 0.0), 0.0)
        self.assertEqual(phi_from_normalized_dispersion(5, 1.0), 1.0)
        self.assertLess(0.0, phi_from_normalized_dispersion(5, 0.2))
        self.assertGreater(1.0, phi_from_normalized_dispersion(5, 0.2))

        rankings = generate_norm_mallows(10, 5, 0.0, random.Random(1))
        self.assertEqual(set(rankings), {(0, 1, 2, 3, 4)})

    def test_generators_and_embedding_shapes(self) -> None:
        rankings = generate_impartial_culture(12, 4, random.Random(4))
        self.assertEqual(len(rankings), 12)
        self.assertTrue(
            all(sorted(ranking) == [0, 1, 2, 3] for ranking in rankings)
        )

        distances = normalized_kendall_distances(np.asarray(rankings))
        coordinates = classical_mds(distances)
        self.assertEqual(coordinates.shape, (12, 2))
        self.assertTrue(np.isfinite(coordinates).all())


if __name__ == "__main__":
    unittest.main()
