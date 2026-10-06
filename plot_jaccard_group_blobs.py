"""Plot voter groups using the partition distance for both clustering and MDS."""

from pathlib import Path
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import hsv_to_rgb
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.spatial import cKDTree

from cluster_candidate_scores import load_input, voter_distances
from plot_jaccard_approver_blobs import approver_region, draw_region
from plot_french_jaccard_mds import metric_mds, normalized_stress


ROOT = Path(__file__).resolve().parent
DEFAULT_COORDINATES = (ROOT / "preference_map_outputs" / "jaccard_mds"
                       / "00026_frenchapproval_jaccard_mds.csv")


def plot_groups(coordinate_file, output_dir, cutoffs, method="complete",
                mds_distance="clustering", election_input=None,
                distance="disagreement", candidate_count=None):
    """Cluster the full input (or saved sample), then draw each cut on one layout."""
    if not cutoffs or any(not np.isfinite(c) or not 0 <= c <= 1 for c in cutoffs):
        raise ValueError("Cutoffs must be finite values in [0,1]")
    if election_input is not None:
        if mds_distance == "saved-jaccard":
            raise ValueError("Saved Jaccard coordinates require sample input")
        names, matrix = load_input(election_input)
        rows = [{"source_input": str(election_input),
                 "approved_candidate_ids": " | ".join(str(c + 1) for c in np.flatnonzero(ballot)),
                 "approved_candidates": " | ".join(names[c] for c in np.flatnonzero(ballot))}
                for ballot in matrix]
        dataset = f"{election_input.stem}_full"
        population_label = "full-election voters"
    else:
        with coordinate_file.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError("Coordinate file contains no voters")
        approvals = [set(row["approved_candidate_ids"].split(" | ")) - {""} for row in rows]
        candidates = sorted(set().union(*approvals), key=int)
        if distance == "hamming" or mds_distance == "hamming":
            if candidate_count is None or candidate_count < max(map(int, candidates), default=1):
                raise ValueError("Sample Hamming distance requires --candidate-count with the full slate size")
            # Retain never-approved candidates in Hamming's denominator.
            candidates = [str(c) for c in range(1, candidate_count + 1)]
        # Never-approved columns have no effect on disagreement distances.
        matrix = np.array([[c in ballot for c in candidates] for ballot in approvals], dtype=int)
        if not candidates:
            matrix = np.zeros((len(rows), 1), dtype=int)
        xy = np.array([(float(row["x"]), float(row["y"])) for row in rows])
        if not np.all(np.isfinite(xy)):
            raise ValueError("MDS coordinates must be finite")
        dataset = coordinate_file.stem.removesuffix("_jaccard_mds")
        population_label = "sampled voters"
    print(f"Computing {distance} distances for {len(rows)} {population_label}...", flush=True)
    distances = voter_distances(matrix, distance)
    embedding_distance = distance if mds_distance == "clustering" else mds_distance
    if embedding_distance in ("disagreement", "hamming", "jaccard"):
        print(f"Fitting MDS with {embedding_distance} distance...", flush=True)
        embedding_distances = distances if embedding_distance == distance else voter_distances(matrix, embedding_distance)
        xy, iterations = metric_mds(embedding_distances)
        stress = normalized_stress(embedding_distances, xy)
        print(f"MDS completed after {iterations} iterations; normalized stress = {stress:.4f}", flush=True)
        rows = [{**row, "x": float(x), "y": float(y)} for row, (x, y) in zip(rows, xy)]
        stem = f"{dataset}_{embedding_distance}_mds"
        embedding_title = f"{embedding_distance}-distance MDS (stress = {stress:.3f})"
    elif mds_distance == "saved-jaccard":
        stem = coordinate_file.stem
        embedding_title = "Jaccard MDS"
    else:
        raise ValueError(f"Unknown MDS distance: {mds_distance}")
    # Different clustering distances on the same embedding need distinct files.
    if distance != ("disagreement" if mds_distance == "saved-jaccard" else embedding_distance):
        stem += f"_{distance}_clustering"
    tree = linkage(squareform(distances, checks=False), method=method) if len(rows) > 1 else None
    output_dir.mkdir(parents=True, exist_ok=True)
    radius = 0.04 * max(float(np.ptp(xy, axis=0).max()), 1e-6)
    limits = [(xy[:, d].min() - 1.5 * radius, xy[:, d].max() + 1.5 * radius) for d in range(2)]
    outputs = []
    for cutoff in cutoffs:
        raw_labels = fcluster(tree, cutoff, criterion="distance") if tree is not None else np.ones(1, dtype=int)
        # Stable group identity: minimum input row belonging to the cluster.
        groups = sorted((np.flatnonzero(raw_labels == g) for g in np.unique(raw_labels)),
                        key=lambda members: (-len(members), int(members[0])))
        fig, ax = plt.subplots(figsize=(13, 9))
        handles = []
        labels = np.empty(len(rows), dtype=int)
        shared = np.zeros(len(rows), dtype=bool)
        # MDS can also leave identical ballots a few floating-point ulps apart.
        for left, right in cKDTree(xy).query_pairs(radius * 1e-7):
            if raw_labels[left] != raw_labels[right]:
                shared[left] = shared[right] = True
        for index, members in enumerate(groups):
            group_id = int(members[0])
            labels[members] = group_id
            # Deterministic color by group identity, including across cutoffs.
            color = hsv_to_rgb(((0.58 + group_id * 0.61803398875) % 1, 0.65, 0.75))
            mask = np.zeros(len(rows), dtype=bool)
            mask[members] = True
            # Coincident points assigned to different clusters cannot be
            # separated in this fixed embedding. Mark them explicitly below.
            region = approver_region(xy[mask & ~shared], xy[~mask], radius)
            draw_region(ax, region, color)
            ax.scatter(*xy[mask].T, s=12, color=color, alpha=0.8, edgecolors="none", zorder=3)
            handles.append(Patch(facecolor=color, alpha=0.6,
                                 label=f"G{group_id}: {len(members)} voters"))
        if shared.any():
            points = np.unique(xy[shared], axis=0)
            ax.scatter(*points.T, marker="x", s=30, color="#111827", linewidths=1.2, zorder=5)
        ax.set(title=f"{dataset}: voter groups on {embedding_title}\n"
                     f"{method.capitalize()} linkage of {distance} distance; cutoff = {cutoff:g}; {len(groups)} groups",
               xlabel="MDS dimension 1", ylabel="MDS dimension 2", xlim=limits[0], ylim=limits[1])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)
        # Keep large partitions readable while drawing every group on the map.
        visible = handles[:24]
        if len(handles) > len(visible):
            visible.append(Patch(facecolor="none", edgecolor="none",
                                 label=f"+ {len(handles) - len(visible)} smaller groups"))
        ax.legend(handles=visible, loc="upper left", bbox_to_anchor=(1.02, 1),
                  frameon=False, fontsize=8, title="Groups (voter counts)")
        fig.text(0.5, 0.02, f"{len(rows)} {population_label}; same MDS coordinates across cutoffs. "
                 + (f"Black crosses: {shared.sum()} voters coincide across groups; no enclosing blob there."
                  if shared.any() else "Group regions cover their members and exclude other groups' points."),
                 ha="center", fontsize=9, color="#475569")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        output = output_dir / f"{stem}_{method}_groups_cutoff_{cutoff:g}.png"
        fig.savefig(output, dpi=230, bbox_inches="tight")
        plt.close(fig)
        with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["voter_id", "group_id", *rows[0]])
            writer.writeheader()
            writer.writerows({"voter_id": i, "group_id": int(labels[i]), **row}
                             for i, row in enumerate(rows))
        print(f"cutoff {cutoff:g}: {len(groups)} groups; sizes {[len(g) for g in groups]} -> {output}", flush=True)
        outputs.append(output)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="Full election: .cat file, directory, or headered approval CSV; default: 00026_frenchapproval")
    source.add_argument("--coordinates", type=Path, help="Instead cluster only the voters in a saved MDS sample")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--distance", choices=("disagreement", "hamming", "jaccard"), default="disagreement",
                        help="Clustering distance; Hamming is normalized to [0,1]")
    parser.add_argument("--candidate-count", type=int, help="Full slate size, required for Hamming on saved samples")
    parser.add_argument("--mds-distance", choices=("clustering", "disagreement", "hamming", "jaccard", "saved-jaccard"),
                        default="clustering", help="Default: use the selected clustering distance for MDS too")
    parser.add_argument("--clustering", choices=("complete", "average"), default="complete")
    parser.add_argument("--cutoffs", type=float, nargs="+", default=[0.9, 0.95, 0.975, 0.99])
    args = parser.parse_args()
    election_input = args.input
    if args.coordinates is None and election_input is None:
        election_input = ROOT / "00026_frenchapproval"
    if election_input is not None and args.mds_distance == "saved-jaccard":
        parser.error("--mds-distance saved-jaccard requires --coordinates")
    coordinates = args.coordinates or DEFAULT_COORDINATES
    embedding_distance = args.distance if args.mds_distance == "clustering" else args.mds_distance
    default_dir = (coordinates.parent / "group_blobs" if args.mds_distance == "saved-jaccard"
                   else coordinates.parent.parent / f"{embedding_distance}_mds" / "group_blobs")
    plot_groups(coordinates, args.output_dir or default_dir,
                args.cutoffs, args.clustering, args.mds_distance, election_input,
                args.distance, args.candidate_count)


if __name__ == "__main__":
    main()
