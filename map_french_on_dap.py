"""Place sampled French approval elections on the IJCAI23 DAP election map.

The published DAP map contains 8-candidate, 96-voter ordinal elections.  This
script pools all voters from the six 00026 districts, retains the eight most
approved candidates in each French dataset, and uses the approval-completion
sampler from :mod:`sample_approval_rankings` to produce one 8x96 election per
dataset.

The original 302 DAP coordinates are kept fixed.  Exact anonymous swap
distances are computed from each French election to well-spread landmark
elections with the C++ extension bundled by the paper.  An out-of-sample k-NN
embedding, calibrated and cross-validated on the original distance table,
places the new elections from their landmark-distance signatures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from map_ordinal_preferences import OrdinalProfile, read_soc
from sample_approval_rankings import (
    ApprovalBallot,
    ApprovalElection,
    parse_preflib_cat,
    sample_rankings,
    write_soc,
)


ROOT = Path(__file__).resolve().parent
DAP_EXPERIMENT = (
    ROOT
    / "diversity-agreement-polarization-IJCAI23"
    / "dap"
    / "dap-code"
    / "experiments"
    / "diversity_map_8_96"
)
DAP_METRICS = (
    ROOT
    / "diversity-agreement-polarization-IJCAI23"
    / "dap"
    / "mapel"
    / "elections"
    / "metrics"
)
DEFAULT_DATASETS = (
    ROOT / "00026_frenchapproval",
    ROOT / "00073_frenchapproval",
)


@dataclass(frozen=True)
class ProjectedElection:
    election_id: str
    election: ApprovalElection
    approval_totals: tuple[int, ...]
    original_candidate_ids: tuple[int, ...]


@dataclass(frozen=True)
class Placement:
    election_id: str
    raw_x: float
    raw_y: float
    plot_x: float
    plot_y: float
    uncertainty: float
    nearest_instances: tuple[str, ...]


def pool_and_project(
    elections: Sequence[ApprovalElection],
    election_id: str,
    title: str,
    num_candidates: int = 8,
) -> ProjectedElection:
    """Pool approval voters and retain candidates with highest approval totals."""
    if not elections:
        raise ValueError("at least one approval election is required")
    candidate_names = elections[0].candidate_names
    if any(election.candidate_names != candidate_names for election in elections[1:]):
        raise ValueError("pooled elections must share the same candidate slate")
    if not 2 <= num_candidates <= len(candidate_names):
        raise ValueError("invalid projected candidate count")

    totals = [0] * len(candidate_names)
    for election in elections:
        for ballot in election.ballots:
            for candidate in ballot.approved:
                totals[candidate] += ballot.count
    # Select by approval total, then restore source order for stable SOC labels.
    selected = sorted(
        sorted(range(len(candidate_names)), key=lambda c: (-totals[c], c))[
            :num_candidates
        ]
    )
    reindex = {old_id: new_id for new_id, old_id in enumerate(selected)}
    projected_counts: Counter[tuple[int, ...]] = Counter()
    for election in elections:
        for ballot in election.ballots:
            projected = tuple(
                reindex[candidate]
                for candidate in ballot.approved
                if candidate in reindex
            )
            projected_counts[projected] += ballot.count
    ballots = tuple(
        ApprovalBallot(count, approved)
        for approved, count in sorted(projected_counts.items())
    )
    pooled = ApprovalElection(
        path=Path(election_id),
        title=title,
        candidate_names=tuple(candidate_names[candidate] for candidate in selected),
        ballots=ballots,
        num_voters=sum(ballot.count for ballot in ballots),
    )
    return ProjectedElection(
        election_id,
        pooled,
        tuple(totals[candidate] for candidate in selected),
        tuple(candidate + 1 for candidate in selected),
    )


def prepare_french_elections(
    datasets: Sequence[Path], output_dir: Path, seed: int
) -> tuple[list[ProjectedElection], list[Path]]:
    """Pool/project each dataset and write its sampled 8x96 SOC election."""
    output_dir.mkdir(parents=True, exist_ok=True)
    projected: list[ProjectedElection] = []
    soc_paths: list[Path] = []
    selection_path = output_dir / "selected_candidates.csv"
    selection_rows: list[tuple[object, ...]] = []

    for index, dataset in enumerate(datasets):
        cat_paths = sorted(Path(dataset).glob("*.cat"))
        if not cat_paths:
            raise FileNotFoundError(f"no .cat files found in {dataset}")
        source_elections = [parse_preflib_cat(path) for path in cat_paths]
        dataset_name = Path(dataset).name
        if dataset_name.startswith("00026"):
            election_id = "French_00026_grouped"
            title = "French 2002 approval (6 districts pooled)"
        elif dataset_name.startswith("00073"):
            election_id = "French_00073"
            title = "French 2017 approval"
        else:
            election_id = f"French_{dataset_name}"
            title = f"French approval ({dataset_name})"
        item = pool_and_project(source_elections, election_id, title)
        projected.append(item)
        file_seed = seed + 10_000 * index
        rankings = sample_rankings(item.election, 96, file_seed, mode="resample")
        soc_path = output_dir / f"{election_id}_8x96.soc"
        write_soc(soc_path, item.election, rankings, file_seed, "resample")
        soc_paths.append(soc_path)
        for projected_id, (original_id, name, total) in enumerate(
            zip(
                item.original_candidate_ids,
                item.election.candidate_names,
                item.approval_totals,
            )
        ):
            selection_rows.append(
                (
                    election_id,
                    projected_id,
                    original_id,
                    name,
                    total,
                    item.election.num_voters,
                )
            )
        print(
            f"{election_id}: pooled {item.election.num_voters} approval voters, "
            f"selected {', '.join(item.election.candidate_names)}, and sampled 96 rankings"
        )

    with selection_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "election_id",
                "projected_candidate_id",
                "original_candidate_id",
                "candidate_name",
                "approval_total",
                "pooled_voters",
            )
        )
        writer.writerows(selection_rows)
    return projected, soc_paths


def load_coordinates(path: Path) -> tuple[list[str], np.ndarray]:
    ids: list[str] = []
    coordinates: list[tuple[float, float]] = []
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source, delimiter=";"):
            ids.append(row["instance_id"])
            coordinates.append((float(row["x"]), float(row["y"])))
    return ids, np.asarray(coordinates, dtype=float)


def load_distance_matrix(path: Path, instance_ids: Sequence[str]) -> np.ndarray:
    indices = {instance_id: index for index, instance_id in enumerate(instance_ids)}
    distances = np.zeros((len(instance_ids), len(instance_ids)), dtype=float)
    observed = np.eye(len(instance_ids), dtype=bool)
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source, delimiter=";"):
            first, second = row["instance_id_1"], row["instance_id_2"]
            if first not in indices or second not in indices:
                continue
            i, j = indices[first], indices[second]
            value = float(row["distance"])
            distances[i, j] = distances[j, i] = value
            observed[i, j] = observed[j, i] = True
    if not observed.all():
        missing = int((~observed).sum() // 2)
        raise ValueError(f"DAP distance table is missing {missing} pairs")
    return distances


def select_landmarks(
    instance_ids: Sequence[str], coordinates: np.ndarray, count: int
) -> list[int]:
    """Select coordinate-spanning landmarks, seeded by the canonical elections."""
    if count < 3 or count > len(instance_ids):
        raise ValueError("landmark count must be between 3 and the number of instances")
    id_to_index = {instance_id: index for index, instance_id in enumerate(instance_ids)}
    selected = [
        id_to_index[instance_id]
        for instance_id in ("ID", "AN", "UN_0")
        if instance_id in id_to_index
    ]
    if not selected:
        selected = [0]
    while len(selected) < count:
        deltas = coordinates[:, None, :] - coordinates[np.asarray(selected)][None, :, :]
        min_squared_distance = np.square(deltas).sum(axis=2).min(axis=1)
        min_squared_distance[np.asarray(selected)] = -1.0
        selected.append(int(np.argmax(min_squared_distance)))
    return selected


def choose_knn_size(
    signatures: np.ndarray, coordinates: np.ndarray
) -> tuple[int, dict[int, float]]:
    """Choose k by leave-one-out median coordinate error on the DAP map."""
    scales = signatures.std(axis=0)
    scales[scales == 0] = 1.0
    normalized = signatures / scales
    candidates = [k for k in (3, 5, 8, 12, 16) if k < len(signatures)]
    errors: dict[int, float] = {}
    for k in candidates:
        point_errors: list[float] = []
        for row in range(len(signatures)):
            feature_distances = np.linalg.norm(normalized - normalized[row], axis=1)
            feature_distances[row] = np.inf
            neighbors = np.argpartition(feature_distances, k)[:k]
            weights = 1.0 / np.maximum(feature_distances[neighbors], 1e-9) ** 2
            prediction = np.average(coordinates[neighbors], axis=0, weights=weights)
            point_errors.append(float(np.linalg.norm(prediction - coordinates[row])))
        errors[k] = float(np.median(point_errors))
    return min(errors, key=errors.get), errors


def place_from_signature(
    signature: np.ndarray,
    training_signatures: np.ndarray,
    coordinates: np.ndarray,
    instance_ids: Sequence[str],
    k: int,
) -> tuple[np.ndarray, float, tuple[str, ...]]:
    scales = training_signatures.std(axis=0)
    scales[scales == 0] = 1.0
    feature_distances = np.linalg.norm(
        (training_signatures - signature) / scales, axis=1
    )
    neighbors = np.argpartition(feature_distances, k)[:k]
    neighbors = neighbors[np.argsort(feature_distances[neighbors])]
    weights = 1.0 / np.maximum(feature_distances[neighbors], 1e-9) ** 2
    point = np.average(coordinates[neighbors], axis=0, weights=weights)
    uncertainty = math.sqrt(
        float(
            np.average(
                np.square(np.linalg.norm(coordinates[neighbors] - point, axis=1)),
                weights=weights,
            )
        )
    )
    return point, uncertainty, tuple(instance_ids[index] for index in neighbors)


def _profile_votes(profile: OrdinalProfile) -> list[list[int]]:
    return [
        list(ranking)
        for count, ranking in profile.counted_rankings
        for _ in range(count)
    ]


def _profile_fingerprint(profile: OrdinalProfile) -> str:
    digest = hashlib.sha256()
    for count, ranking in profile.counted_rankings:
        digest.update(f"{count}:{','.join(map(str, ranking))};".encode("ascii"))
    return digest.hexdigest()[:16]


def load_swap_distance() -> Callable[[list[list[int]], list[list[int]]], float]:
    extensions = sorted(DAP_METRICS.glob("cppdistances*.pyd"))
    extensions += sorted(DAP_METRICS.glob("cppdistances*.so"))
    if not extensions:
        raise FileNotFoundError(
            f"no compatible cppdistances extension found in {DAP_METRICS}"
        )
    specification = importlib.util.spec_from_file_location("cppdistances", extensions[0])
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {extensions[0]}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.swapd


def compute_landmark_signatures(
    french_profiles: Sequence[OrdinalProfile],
    landmark_ids: Sequence[str],
    elections_dir: Path,
    cache_path: Path,
) -> dict[str, np.ndarray]:
    """Compute or reuse exact swap distances from French profiles to landmarks."""
    cached: dict[tuple[str, str, str], float] = {}
    if cache_path.exists():
        with cache_path.open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                cached[(row["election_id"], row["fingerprint"], row["landmark_id"])] = float(
                    row["distance"]
                )
    swap_distance = load_swap_distance()
    landmark_votes = {
        landmark_id: _profile_votes(read_soc(elections_dir / f"{landmark_id}.soc"))
        for landmark_id in landmark_ids
    }
    rows: list[tuple[object, ...]] = []
    signatures: dict[str, np.ndarray] = {}
    for profile in french_profiles:
        fingerprint = _profile_fingerprint(profile)
        votes = _profile_votes(profile)
        if len(votes) != 96 or any(len(vote) != 8 for vote in votes):
            raise ValueError(f"{profile.path}: DAP placement requires an 8x96 election")
        values: list[float] = []
        for landmark_number, landmark_id in enumerate(landmark_ids, start=1):
            key = (profile.path.stem, fingerprint, landmark_id)
            started = time.perf_counter()
            if key in cached:
                distance = cached[key]
                elapsed = 0.0
                status = "cached"
            else:
                distance = float(swap_distance(votes, landmark_votes[landmark_id]))
                elapsed = time.perf_counter() - started
                status = "computed"
            values.append(distance)
            rows.append(
                (profile.path.stem, fingerprint, landmark_id, distance, elapsed)
            )
            print(
                f"  {profile.path.stem} -> {landmark_id} "
                f"({landmark_number}/{len(landmark_ids)}): {distance:g} "
                f"[{status}, {elapsed:.2f}s]"
            )
        signatures[profile.path.stem] = np.asarray(values, dtype=float)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("election_id", "fingerprint", "landmark_id", "distance", "seconds"))
        writer.writerows(rows)
    return signatures


def orient_coordinates(
    coordinates: np.ndarray, id_point: np.ndarray
) -> np.ndarray:
    """Apply the reflection/rotation used by diversity_map_8_96.py."""
    reflected = coordinates.copy()
    reflected[:, 1] *= -1.0
    pivot = np.asarray((id_point[0], -id_point[1]), dtype=float)
    angle = -math.pi * 0.825
    rotation = np.asarray(
        ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle)))
    )
    return (reflected - pivot) @ rotation.T + pivot


def _family_for_instance(instance_id: str, family_rows: Sequence[dict[str, str]]) -> dict[str, str]:
    for row in family_rows:
        family_id = row["family_id"]
        size = int(row["size"])
        if (size == 1 and instance_id == family_id) or (
            size > 1 and instance_id.startswith(f"{family_id}_")
        ):
            suffix = instance_id[len(family_id) + 1 :] if size > 1 else "0"
            if size == 1 or suffix.isdigit():
                return row
    return {
        "family_id": "Other",
        "label": "Other",
        "color": "gray",
        "marker": "o",
        "alpha": "0.5",
    }


def plot_dap_map(
    instance_ids: Sequence[str],
    raw_coordinates: np.ndarray,
    french_raw: dict[str, np.ndarray],
    map_csv: Path,
    output_path: Path,
) -> dict[str, np.ndarray]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    id_index = instance_ids.index("ID")
    combined = np.vstack((raw_coordinates, *french_raw.values()))
    oriented = orient_coordinates(combined, raw_coordinates[id_index])
    base_oriented = oriented[: len(raw_coordinates)]
    french_oriented = oriented[len(raw_coordinates) :]
    with map_csv.open(encoding="utf-8", newline="") as source:
        family_rows = list(csv.DictReader(source, delimiter=";"))

    figure, axis = plt.subplots(figsize=(11.5, 8.3))
    plotted_labels: set[str] = set()
    for index, instance_id in enumerate(instance_ids):
        family = _family_for_instance(instance_id, family_rows)
        label = family["label"]
        legend_label = label if label not in plotted_labels else "_nolegend_"
        plotted_labels.add(label)
        axis.scatter(
            base_oriented[index, 0],
            base_oriented[index, 1],
            color=family["color"],
            marker=family["marker"],
            alpha=float(family["alpha"]),
            s=25,
            label=legend_label,
            zorder=2,
        )

    french_styles = (
        ("#e41a1c", "*", "French 2002 (6 districts pooled)"),
        ("#377eb8", "P", "French 2017"),
    )
    oriented_lookup: dict[str, np.ndarray] = {}
    for (election_id, _), point, (color, marker, label) in zip(
        french_raw.items(), french_oriented, french_styles
    ):
        oriented_lookup[election_id] = point
        axis.scatter(
            point[0],
            point[1],
            color=color,
            marker=marker,
            s=240,
            edgecolors="black",
            linewidths=0.9,
            label=label,
            zorder=6,
        )
        axis.annotate(
            label,
            point,
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            zorder=7,
        )
    axis.set_title("French approval elections on the IJCAI23 DAP map")
    axis.set_xlabel("MDS dimension 1")
    axis.set_ylabel("MDS dimension 2")
    axis.grid(alpha=0.16)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, ncol=1)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return oriented_lookup


def run_pipeline(
    datasets: Sequence[Path] = DEFAULT_DATASETS,
    output_dir: Path = ROOT / "preference_map_outputs" / "dap",
    seed: int = 23,
    num_landmarks: int = 20,
) -> tuple[Path, Path]:
    projected, soc_paths = prepare_french_elections(datasets, output_dir, seed)
    coordinate_path = DAP_EXPERIMENT / "coordinates" / "mds_swap_2d.csv"
    distance_path = DAP_EXPERIMENT / "distances" / "swap.csv"
    elections_dir = DAP_EXPERIMENT / "elections"
    instance_ids, coordinates = load_coordinates(coordinate_path)
    distances = load_distance_matrix(distance_path, instance_ids)
    landmark_indices = select_landmarks(instance_ids, coordinates, num_landmarks)
    landmark_ids = [instance_ids[index] for index in landmark_indices]
    training_signatures = distances[:, landmark_indices]
    k, cv_errors = choose_knn_size(training_signatures, coordinates)
    print(
        "Landmark-signature leave-one-out median errors: "
        + ", ".join(f"k={key}: {value:.1f}" for key, value in cv_errors.items())
    )
    print(f"Using k={k} and landmarks: {', '.join(landmark_ids)}")

    french_profiles = [read_soc(path) for path in soc_paths]
    signatures = compute_landmark_signatures(
        french_profiles,
        landmark_ids,
        elections_dir,
        output_dir / "landmark_distances.csv",
    )
    french_raw: dict[str, np.ndarray] = {}
    placement_details: dict[str, tuple[float, tuple[str, ...]]] = {}
    for profile in french_profiles:
        point, uncertainty, nearest = place_from_signature(
            signatures[profile.path.stem],
            training_signatures,
            coordinates,
            instance_ids,
            k,
        )
        french_raw[profile.path.stem] = point
        placement_details[profile.path.stem] = (uncertainty, nearest)
        print(
            f"{profile.path.stem}: raw placement ({point[0]:.2f}, {point[1]:.2f}); "
            f"nearest DAP elections: {', '.join(nearest[:5])}"
        )

    map_path = output_dir / "french_on_dap_map.png"
    oriented = plot_dap_map(
        instance_ids,
        coordinates,
        french_raw,
        DAP_EXPERIMENT / "map.csv",
        map_path,
    )
    placements_path = output_dir / "french_dap_placements.csv"
    with placements_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "election_id",
                "raw_x",
                "raw_y",
                "plot_x",
                "plot_y",
                "local_uncertainty",
                "knn_k",
                "cv_median_error",
                "nearest_dap_instances",
            )
        )
        for election_id, raw_point in french_raw.items():
            uncertainty, nearest = placement_details[election_id]
            plot_point = oriented[election_id]
            writer.writerow(
                (
                    election_id,
                    raw_point[0],
                    raw_point[1],
                    plot_point[0],
                    plot_point[1],
                    uncertainty,
                    k,
                    cv_errors[k],
                    " | ".join(nearest),
                )
            )
    print(f"Wrote DAP map: {map_path}")
    print(f"Wrote placement details: {placements_path}")
    return map_path, placements_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample the French approval elections and place them on the IJCAI23 DAP map."
    )
    parser.add_argument("--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "preference_map_outputs" / "dap"
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--num-landmarks", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(args.datasets, args.output_dir, args.seed, args.num_landmarks)


if __name__ == "__main__":
    main()
