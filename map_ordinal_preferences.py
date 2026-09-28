"""Map sampled ordinal preferences beside IJCAI23 benchmark cultures.

The map uses normalized Kendall (swap) distances between strict rankings and a
two-dimensional classical MDS embedding.  This is the same ranking distance and
the same broad mapping idea used by the bundled IJCAI23 preference-map code,
while keeping this script independent of the repository's old Mapel build.
"""

from __future__ import annotations

import argparse
import ast
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class OrdinalProfile:
    path: Path
    title: str
    candidate_names: tuple[str, ...]
    counted_rankings: tuple[tuple[int, tuple[int, ...]], ...]

    @property
    def num_rankings(self) -> int:
        return sum(count for count, _ in self.counted_rankings)


@dataclass(frozen=True)
class MapGroup:
    label: str
    source: str
    rankings: tuple[tuple[int, ...], ...]


def read_soc(path: Path) -> OrdinalProfile:
    """Read an aggregated or unaggregated PrefLib/Mapel strict-order file."""
    path = Path(path)
    with path.open(encoding="utf-8-sig") as source:
        lines = [line.rstrip("\r\n") for line in source if line.strip()]
    if not lines:
        raise ValueError(f"{path}: empty file")

    cursor = 0
    metadata: dict[object, object] = {}
    model_name = "ordinal"
    if lines[cursor].lstrip().startswith("#"):
        comment = lines[cursor].lstrip()[1:].strip()
        model_name, separator, params_text = comment.partition(" ")
        if separator and params_text.lstrip().startswith("{"):
            try:
                parsed = ast.literal_eval(params_text)
                if isinstance(parsed, dict):
                    metadata = parsed
            except (SyntaxError, ValueError):
                metadata = {}
        cursor += 1

    try:
        num_candidates = int(lines[cursor].strip())
    except (IndexError, ValueError) as error:
        raise ValueError(f"{path}: missing candidate count") from error
    cursor += 1
    if num_candidates < 2:
        raise ValueError(f"{path}: at least two candidates are required")

    candidate_names: list[str] = []
    candidate_ids: list[int] = []
    for _ in range(num_candidates):
        try:
            id_text, name = lines[cursor].split(",", 1)
            candidate_ids.append(int(id_text.strip()))
            candidate_names.append(name.strip())
        except (IndexError, ValueError) as error:
            raise ValueError(f"{path}: malformed candidate line") from error
        cursor += 1
    if len(set(candidate_ids)) != num_candidates:
        raise ValueError(f"{path}: candidate IDs are not unique")
    id_to_index = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}

    try:
        summary = [int(value.strip()) for value in lines[cursor].split(",")]
        declared_voters = summary[0]
        declared_orders = summary[2]
    except (IndexError, ValueError) as error:
        raise ValueError(f"{path}: malformed election summary") from error
    cursor += 1

    counted_rankings: list[tuple[int, tuple[int, ...]]] = []
    for order_index in range(declared_orders):
        try:
            values = [int(value.strip()) for value in lines[cursor].split(",")]
        except (IndexError, ValueError) as error:
            raise ValueError(f"{path}: malformed ranking {order_index + 1}") from error
        cursor += 1
        count, raw_ranking = values[0], values[1:]
        if count <= 0 or len(raw_ranking) != num_candidates:
            raise ValueError(f"{path}: malformed ranking {order_index + 1}")
        try:
            ranking = tuple(id_to_index[candidate_id] for candidate_id in raw_ranking)
        except KeyError as error:
            raise ValueError(f"{path}: unknown candidate ID {error.args[0]}") from error
        if len(set(ranking)) != num_candidates:
            raise ValueError(f"{path}: ranking {order_index + 1} is not a permutation")
        counted_rankings.append((count, ranking))

    observed_voters = sum(count for count, _ in counted_rankings)
    if observed_voters != declared_voters:
        raise ValueError(
            f"{path}: ranking counts total {observed_voters}, "
            f"but the summary says {declared_voters}"
        )
    title = str(metadata.get("title") or path.stem)
    return OrdinalProfile(
        path, title, tuple(candidate_names), tuple(counted_rankings)
    )


def _sample_profile(
    profile: OrdinalProfile, maximum: int, rng: random.Random
) -> tuple[tuple[int, ...], ...]:
    expanded = [
        ranking
        for count, ranking in profile.counted_rankings
        for _ in range(count)
    ]
    if len(expanded) > maximum:
        expanded = rng.sample(expanded, maximum)
    return tuple(expanded)


def generate_impartial_culture(
    num_rankings: int, num_candidates: int, rng: random.Random
) -> tuple[tuple[int, ...], ...]:
    rankings: list[tuple[int, ...]] = []
    for _ in range(num_rankings):
        ranking = list(range(num_candidates))
        rng.shuffle(ranking)
        rankings.append(tuple(ranking))
    return tuple(rankings)


def _expected_mallows_inversions(num_candidates: int, phi: float) -> float:
    expected = 0.0
    for inserted_index in range(1, num_candidates):
        weights = [phi**inversions for inversions in range(inserted_index + 1)]
        normalizer = sum(weights)
        expected += sum(
            inversions * weight for inversions, weight in enumerate(weights)
        ) / normalizer
    return expected


def phi_from_normalized_dispersion(num_candidates: int, norm_phi: float) -> float:
    """Convert the IJCAI/Mapel normalized Mallows parameter to ordinary phi."""
    if not 0.0 <= norm_phi <= 1.0:
        raise ValueError("normalized Mallows dispersion must lie in [0, 1]")
    if norm_phi == 0.0:
        return 0.0
    if norm_phi == 1.0:
        return 1.0
    target = norm_phi * num_candidates * (num_candidates - 1) / 4.0
    low, high = 0.0, 1.0
    for _ in range(70):
        middle = (low + high) / 2.0
        if _expected_mallows_inversions(num_candidates, middle) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def generate_norm_mallows(
    num_rankings: int,
    num_candidates: int,
    norm_phi: float,
    rng: random.Random,
) -> tuple[tuple[int, ...], ...]:
    """Generate a repeated-insertion Mallows election centered at 0,1,...,m-1."""
    phi = phi_from_normalized_dispersion(num_candidates, norm_phi)
    rankings: list[tuple[int, ...]] = []
    for _ in range(num_rankings):
        ranking = [0]
        for candidate in range(1, num_candidates):
            positions = list(range(candidate + 1))
            weights = [phi ** (candidate - position) for position in positions]
            position = rng.choices(positions, weights=weights, k=1)[0]
            ranking.insert(position, candidate)
        rankings.append(tuple(ranking))
    return tuple(rankings)


def normalized_kendall_distances(rankings: np.ndarray) -> np.ndarray:
    """Return all-pairs normalized Kendall distances for permutation rows."""
    if rankings.ndim != 2:
        raise ValueError("rankings must be a two-dimensional array")
    num_rankings, num_candidates = rankings.shape
    positions = np.empty_like(rankings)
    positions[np.arange(num_rankings)[:, None], rankings] = np.arange(num_candidates)
    first, second = np.triu_indices(num_candidates, k=1)
    pairwise_preferences = positions[:, first] < positions[:, second]
    num_pairs = pairwise_preferences.shape[1]
    distances = np.empty((num_rankings, num_rankings), dtype=float)
    for row in range(num_rankings):
        distances[row] = np.count_nonzero(
            pairwise_preferences != pairwise_preferences[row], axis=1
        ) / num_pairs
    return distances


def classical_mds(distances: np.ndarray, dimensions: int = 2) -> np.ndarray:
    """Compute a deterministic classical multidimensional-scaling embedding."""
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be a square matrix")
    squared = distances * distances
    row_means = squared.mean(axis=1, keepdims=True)
    column_means = squared.mean(axis=0, keepdims=True)
    grand_mean = squared.mean()
    gram = -0.5 * (squared - row_means - column_means + grand_mean)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    chosen = np.argsort(eigenvalues)[::-1][:dimensions]
    positive = np.maximum(eigenvalues[chosen], 0.0)
    coordinates = eigenvectors[:, chosen] * np.sqrt(positive)
    if coordinates.shape[1] < dimensions:
        coordinates = np.pad(
            coordinates, ((0, 0), (0, dimensions - coordinates.shape[1]))
        )
    return coordinates


def build_groups(
    profiles: Sequence[OrdinalProfile],
    max_per_source: int,
    seed: int,
    cultures: Sequence[str],
    mallows_norm_phi: float,
) -> list[MapGroup]:
    if not profiles:
        raise ValueError("at least one ordinal profile is required")
    if max_per_source <= 0:
        raise ValueError("max_per_source must be positive")
    reference_names = profiles[0].candidate_names
    for profile in profiles[1:]:
        if profile.candidate_names != reference_names:
            raise ValueError(
                "all profiles in one map must have the same candidate names and order"
            )

    rng = random.Random(seed)
    groups: list[MapGroup] = []
    for profile in profiles:
        rankings = _sample_profile(profile, max_per_source, rng)
        groups.append(MapGroup(profile.title, profile.path.name, rankings))

    synthetic_size = min(max_per_source, max(len(group.rankings) for group in groups))
    num_candidates = len(reference_names)
    for culture in cultures:
        if culture == "impartial":
            rankings = generate_impartial_culture(synthetic_size, num_candidates, rng)
            groups.append(MapGroup("Impartial culture", "synthetic:impartial", rankings))
        elif culture == "mallows":
            rankings = generate_norm_mallows(
                synthetic_size, num_candidates, mallows_norm_phi, rng
            )
            groups.append(
                MapGroup(
                    f"Norm-Mallows {mallows_norm_phi:g}",
                    f"synthetic:norm-mallows-{mallows_norm_phi:g}",
                    rankings,
                )
            )
        else:
            raise ValueError(f"unknown culture: {culture}")
    return groups


def write_coordinates(
    path: Path, groups: Sequence[MapGroup], coordinates: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("source", "source_file", "ranking", "x", "y"))
        coordinate_index = 0
        for group in groups:
            for ranking in group.rankings:
                x, y = coordinates[coordinate_index]
                writer.writerow(
                    (group.label, group.source, " > ".join(map(str, ranking)), x, y)
                )
                coordinate_index += 1


def plot_map(
    groups: Sequence[MapGroup],
    coordinates: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    # Import lazily so parsing/generation utilities can be used without plotting.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 7.5))
    palette = plt.get_cmap("tab10")
    coordinate_index = 0
    for group_index, group in enumerate(groups):
        group_size = len(group.rankings)
        group_coordinates = coordinates[coordinate_index : coordinate_index + group_size]
        coordinate_index += group_size
        is_synthetic = group.source.startswith("synthetic:")
        style = {}
        if not is_synthetic:
            style = {"edgecolors": "white"}
        axis.scatter(
            group_coordinates[:, 0],
            group_coordinates[:, 1],
            s=28 if is_synthetic else 34,
            alpha=0.62 if is_synthetic else 0.78,
            marker="x" if is_synthetic else "o",
            linewidths=1.0 if is_synthetic else 0.35,
            color=palette(group_index % 10),
            label=f"{group.label} (n={group_size})",
            **style,
        )
    axis.set_title(title)
    axis.set_xlabel("MDS dimension 1")
    axis.set_ylabel("MDS dimension 2")
    axis.grid(alpha=0.18)
    axis.legend(loc="best", fontsize=8, frameon=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def map_soc_files(
    inputs: Sequence[Path],
    output_path: Path,
    max_per_source: int = 75,
    seed: int = 23,
    cultures: Sequence[str] = ("impartial", "mallows"),
    mallows_norm_phi: float = 0.2,
    title: Optional[str] = None,
) -> tuple[Path, Path]:
    profiles = [read_soc(path) for path in inputs]
    groups = build_groups(profiles, max_per_source, seed, cultures, mallows_norm_phi)
    all_rankings = np.asarray(
        [ranking for group in groups for ranking in group.rankings], dtype=int
    )
    print(f"Computing Kendall distances for {len(all_rankings)} rankings ...")
    distances = normalized_kendall_distances(all_rankings)
    print("Embedding rankings in two dimensions ...")
    coordinates = classical_mds(distances)
    output_path = Path(output_path)
    coordinate_path = output_path.with_suffix(".csv")
    write_coordinates(coordinate_path, groups, coordinates)
    plot_map(
        groups,
        coordinates,
        output_path,
        title or "Approval-consistent rankings and IJCAI23 cultures",
    )
    print(f"Wrote map: {output_path}")
    print(f"Wrote coordinates: {coordinate_path}")
    return output_path, coordinate_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Kendall-distance preference map from sampled SOC elections "
            "and IJCAI23 benchmark cultures."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Sampled .soc files")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("preference_map_outputs") / "preference_map.png",
        help="Output PNG path (a coordinate CSV is written beside it)",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=75,
        help="Maximum plotted rankings from each real or synthetic source",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--cultures",
        nargs="+",
        choices=("impartial", "mallows"),
        default=("impartial", "mallows"),
    )
    parser.add_argument(
        "--mallows-norm-phi",
        type=float,
        default=0.2,
        help="Normalized dispersion for the Mallows benchmark",
    )
    parser.add_argument("--title", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    map_soc_files(
        args.inputs,
        args.output,
        max_per_source=args.max_per_source,
        seed=args.seed,
        cultures=args.cultures,
        mallows_norm_phi=args.mallows_norm_phi,
        title=args.title,
    )


if __name__ == "__main__":
    main()
