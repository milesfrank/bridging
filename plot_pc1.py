"""Project voters onto PC1 of the approval matrix and plot a smoothed density.

Columns are mean-centered, then the first right singular vector is the
principal component. The sign is flipped so Jospin loads positive. The
curve is a histogram of PC1 scores with a light Gaussian smooth.

Pass --candidate NAME to restrict the curve to voters who approved that
candidate. PC1 itself is always estimated from the full matrix.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "matrices" / "frenchapproval.csv"


def load_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        names = next(reader)
        X = np.array([[int(v) for v in row] for row in reader], dtype=float)
    return names, X


def pc1_scores(X: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray, float]:
    """Return PC1 scores, loadings, and fraction of variance explained."""
    Xc = X - X.mean(axis=0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    loadings = Vt[0]
    scores = Xc @ loadings
    if "Jospin" in names and "LePen" in names:
        if loadings[names.index("LePen")] > loadings[names.index("Jospin")]:
            loadings = -loadings
            scores = -scores
    var_frac = float((S[0] ** 2) / np.sum(S ** 2))
    return scores, loadings, var_frac


def resolve_candidate(names: list[str], query: str) -> str:
    folded = {n.casefold(): n for n in names}
    key = query.casefold().replace(" ", "").replace("-", "")
    compact = {n.casefold().replace(" ", "").replace("-", ""): n for n in names}
    if query in names:
        return query
    if query.casefold() in folded:
        return folded[query.casefold()]
    if key in compact:
        return compact[key]
    listed = ", ".join(names)
    raise SystemExit(f"Unknown candidate {query!r}. Choose one of: {listed}")


def smoothed_density(
    scores: np.ndarray,
    bins: int = 80,
    sigma: float = 1.4,
    score_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Density histogram, then a light Gaussian blur along the bin axis."""
    density, edges = np.histogram(
        scores, bins=bins, range=score_range, density=True
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(density, kernel, mode="same")
    return centers, smooth


def plot_pc1_line(
    scores: np.ndarray,
    var_frac: float,
    out_path: Path,
    bins: int,
    sigma: float,
    show: bool,
    candidate: str | None = None,
    n_total: int | None = None,
    score_range: tuple[float, float] | None = None,
) -> None:
    x, y = smoothed_density(
        scores, bins=bins, sigma=sigma, score_range=score_range
    )
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(x, y, color="#1f4e79", linewidth=2)
    ax.fill_between(x, y, color="#1f4e79", alpha=0.12)
    ax.axvline(0, color="0.65", linewidth=0.8, linestyle="--")
    ax.set_xlabel("PC1 score")
    ax.set_ylabel("Density (smoothed)")
    if candidate is None:
        title = (
            f"Voters along PC1  ·  {len(scores)} ballots  ·  "
            f"{var_frac:.1%} of variance  ·  Gaussian σ={sigma:g} bins"
        )
    else:
        denom = n_total if n_total else len(scores)
        title = (
            f"{candidate} approvers along PC1  ·  "
            f"{len(scores)} of {denom} voters  ·  "
            f"Gaussian σ={sigma:g} bins"
        )
    ax.set_title(title)
    ax.text(
        0.02,
        0.94,
        "Chirac / right-leaning",
        transform=ax.transAxes,
        fontsize=9,
        color="0.35",
        va="top",
    )
    ax.text(
        0.98,
        0.94,
        "Jospin / left-leaning",
        transform=ax.transAxes,
        fontsize=9,
        color="0.35",
        va="top",
        ha="right",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f"wrote {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a slightly smoothed density of voters on PC1."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "matrices" / "pc1_density.png",
    )
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.4,
        help="Gaussian smooth width in histogram bins (slight ≈ 1–2)",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--candidate",
        type=str,
        default=None,
        help="If set, only plot voters who approved this candidate "
        "(PC1 is still fit on everyone)",
    )
    args = parser.parse_args()

    names, X = load_matrix(args.csv)
    scores, loadings, var_frac = pc1_scores(X, names)
    print(f"{X.shape[0]} voters x {X.shape[1]} candidates")
    print(f"PC1 variance explained: {var_frac:.1%}")
    order = np.argsort(loadings)
    print("PC1 loadings:")
    for i in order:
        print(f"  {names[i]:15s} {loadings[i]:8.4f}")

    candidate = None
    plotted = scores
    out_path = args.out
    if args.candidate:
        candidate = resolve_candidate(names, args.candidate)
        col = names.index(candidate)
        mask = X[:, col] == 1
        plotted = scores[mask]
        print(f"{candidate} approvers: {int(mask.sum())} / {len(scores)}")
        if plotted.size == 0:
            raise SystemExit(f"No voters approved {candidate}")
        if args.out == ROOT / "matrices" / "pc1_density.png":
            safe = candidate.replace(" ", "_")
            out_path = ROOT / "matrices" / f"pc1_{safe}_approvers.png"

    plot_pc1_line(
        plotted,
        var_frac,
        out_path,
        args.bins,
        args.sigma,
        args.show,
        candidate=candidate,
        n_total=len(scores),
        score_range=(float(scores.min()), float(scores.max())),
    )


if __name__ == "__main__":
    main()
