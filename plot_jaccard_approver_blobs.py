"""Cover approvers with unions of disks that exclude every disapprover.

Requires numpy, scipy, matplotlib, and shapely. Uses the saved coordinates
from saved ballots. Polygon holes are preserved in the rendered regions.
"""

from pathlib import Path
import argparse
import csv
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as PlotPath
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from shapely.geometry import Point
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from plot_french_jaccard_mds import jaccard_distances, metric_mds, normalized_stress


def approver_region(points, disapprovers, maximum_radius):
    """Merge safe disks; each radius is below the nearest disapprover distance."""
    points = np.unique(points, axis=0)
    if len(disapprovers):
        nearest = cKDTree(disapprovers).query(points)[0]
        if np.any(nearest == 0):
            raise ValueError("An approver and disapprover coincide: exact separation is impossible.")
        radii = np.minimum(maximum_radius, 0.8 * nearest)
    else:
        radii = np.full(len(points), maximum_radius)
    region = unary_union([Point(point).buffer(radius, resolution=32)
                          for point, radius in zip(points, radii)])
    if not all(region.contains(Point(point)) for point in points):
        raise AssertionError("An approver is not covered")
    if any(region.intersects(Point(point)) for point in disapprovers):
        raise AssertionError("A disapprover touches or lies inside a region")
    return region


def region_path(polygon):
    """Opposite ring orientations leave holes unfilled in Matplotlib."""
    polygon = orient(polygon, sign=1.0)
    paths = []
    for ring in [polygon.exterior, *polygon.interiors]:
        vertices = np.asarray(ring.coords)
        codes = [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(vertices) - 2) + [PlotPath.CLOSEPOLY]
        paths.append(PlotPath(vertices, codes))
    return PlotPath.make_compound_path(*paths)


def draw_region(axis, region, color):
    polygons = [region] if region.geom_type == "Polygon" else list(region.geoms)
    for polygon in polygons:
        axis.add_patch(PathPatch(region_path(polygon), facecolor=to_rgba(color, 0.18),
                                 edgecolor=color, linewidth=1.0, zorder=2))
    return len(polygons)


def plot_individual_candidates(folder, stem, xy, approvals, names, stress, maximum_radius, metric):
    output_dir = folder / f"{stem}_by_candidate"
    output_dir.mkdir(parents=True, exist_ok=True)
    xlim = (xy[:, 0].min() - maximum_radius * 1.5,
            xy[:, 0].max() + maximum_radius * 1.5)
    ylim = (xy[:, 1].min() - maximum_radius * 1.5,
            xy[:, 1].max() + maximum_radius * 1.5)
    palette = plt.get_cmap("tab20")
    for index, name in enumerate(names):
        mask = np.array([name in ballot for ballot in approvals])
        approvers, disapprovers = xy[mask], xy[~mask]
        region = approver_region(approvers, disapprovers, maximum_radius)
        color = palette(index)
        fig, ax = plt.subplots(figsize=(10, 8))
        blob_count = draw_region(ax, region, color)
        ax.scatter(*disapprovers.T, s=10, c="#94a3b8", alpha=0.45,
                   edgecolors="none", label=f"Disapprovers ({len(disapprovers)})", zorder=3)
        ax.scatter(*approvers.T, s=16, c=[color], alpha=0.85,
                   edgecolors="none", label=f"Approvers ({len(approvers)})", zorder=4)
        ax.set(title=f"00026: {name} approvers on {metric.capitalize()} MDS\n"
                     f"{blob_count} regions; normalized stress = {stress:.3f}",
               xlabel="MDS dimension 1", ylabel="MDS dimension 2",
               xlim=xlim, ylim=ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)
        ax.legend(loc="upper right", frameon=True)
        fig.tight_layout()
        safe_name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        output = output_dir / f"{safe_name}_approver_blobs.png"
        fig.savefig(output, dpi=230, bbox_inches="tight")
        plt.close(fig)
        print(f"{name}: {blob_count} regions; {len(approvers)} approvers covered; "
              f"0 disapprovers in regions -> {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=("jaccard", "hamming"), default="jaccard",
                        help="MDS distance; Hamming counts differing approval bits")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / "preference_map_outputs"
    original_csv = root / "jaccard_mds/00026_frenchapproval_jaccard_mds.csv"
    with original_csv.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    approvals = [set(row["approved_candidates"].split(" | ")) - {""} for row in rows]
    names = sorted(set().union(*approvals))
    matrix = np.array([[name in ballot for name in names] for ballot in approvals], dtype=np.int8)
    if args.metric == "hamming":
        # One bit per candidate; cityblock gives the count of differing approvals.
        distances = cdist(matrix, matrix, metric="cityblock")
        folder = root / "hamming_mds"
        folder.mkdir(parents=True, exist_ok=True)
        stem = "00026_frenchapproval_hamming_mds"
        coordinate_file = folder / f"{stem}.csv"
        if coordinate_file.exists():
            with coordinate_file.open(encoding="utf-8", newline="") as source:
                saved = list(csv.DictReader(source))
            if len(saved) != len(rows) or any(
                tuple(row[k] for k in ("source_file", "approved_candidate_ids")) !=
                tuple(old[k] for k in ("source_file", "approved_candidate_ids"))
                for row, old in zip(saved, rows)
            ):
                raise ValueError("Saved Hamming coordinates do not match the source ballots")
            xy = np.array([(float(row["x"]), float(row["y"])) for row in saved])
        else:
            print(f"Computing Hamming MDS for {len(rows)} ballots...", flush=True)
            xy, iterations = metric_mds(distances)
            with coordinate_file.open("w", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=rows[0].keys())
                writer.writeheader()
                for row, (x, y) in zip(rows, xy):
                    writer.writerow({**row, "x": x, "y": y})
            print(f"MDS finished after {iterations} iterations; saved {coordinate_file}", flush=True)
    else:
        folder = root / "jaccard_mds"
        stem = "00026_frenchapproval_jaccard_mds"
        xy = np.array([(float(row["x"]), float(row["y"])) for row in rows])
        distances = jaccard_distances(matrix)
    stress = normalized_stress(distances, xy)
    candidates = [
        "Chevenement", 
        "Jospin", 
        "Bayrou", 
        "Chirac", 
        "Lepage"
    ]
    colors = [
        "#8e44ad", 
        "#d43f4f", 
        "#dd9200", 
        "#2474b5", 
        "#18966a"
    ]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(*xy.T, s=7, color="#475569", alpha=0.5, edgecolors="none", zorder=3)
    maximum_radius = 0.04 * np.ptp(xy, axis=0).max()
    handles = []
    for name, color in zip(candidates, colors):
        mask = np.array([name in ballot for ballot in approvals])
        points = xy[mask]
        region = approver_region(points, xy[~mask], maximum_radius)
        polygons = [region] if region.geom_type == "Polygon" else list(region.geoms)
        for polygon in polygons:
            ax.add_patch(PathPatch(region_path(polygon), facecolor=to_rgba(color, 0.13),
                                   edgecolor=color, linewidth=0.8, zorder=2))
        handles.append(Patch(facecolor=to_rgba(color, 0.13), edgecolor=color,
                             label=f"{name} ({len(points)} approvers)"))
        print(f"{name}: {len(polygons)} blobs; {len(points)}/{len(points)} approvers covered; 0 disapprovers")
    ax.set(title=f"00026: approval ballots by {args.metric.capitalize()}-distance MDS\n"
                 f"Approver-only regions; normalized stress = {stress:.3f}",
           xlabel="MDS dimension 1", ylabel="MDS dimension 2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False,
              title="Approvers of")
    fig.text(0.5, 0.02, "Each candidate's regions cover all their sampled approver points and exclude all their disapprover points.",
             ha="center", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for extension in ("png", "svg"):
        output = folder / f"{stem}_approver_blobs.{extension}"
        fig.savefig(output, dpi=230, bbox_inches="tight")
        print(output)
    plt.close(fig)
    plot_individual_candidates(folder, stem, xy, approvals, names, stress, maximum_radius,
                               args.metric)


if __name__ == "__main__":
    main()
