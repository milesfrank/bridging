"""Sample both French approval datasets and create comparable preference maps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from map_ordinal_preferences import map_soc_files
from sample_approval_rankings import sample_files


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = (
    REPOSITORY_ROOT / "00026_frenchapproval",
    REPOSITORY_ROOT / "00073_frenchapproval",
)


def run_pipeline(
    datasets: Sequence[Path],
    output_dir: Path,
    num_rankings: int = 250,
    map_per_source: int = 75,
    seed: int = 23,
    mallows_norm_phi: float = 0.2,
) -> list[tuple[Path, Path]]:
    """Run sampling followed by mapping, once for each candidate slate."""
    results: list[tuple[Path, Path]] = []
    for dataset_index, dataset in enumerate(datasets):
        dataset = Path(dataset)
        if not dataset.is_dir():
            raise FileNotFoundError(f"dataset directory not found: {dataset}")
        dataset_seed = seed + 10_000 * dataset_index
        sample_dir = Path(output_dir) / "ordinal_samples" / dataset.name
        soc_files = sample_files(
            [dataset],
            sample_dir,
            num_rankings=num_rankings,
            seed=dataset_seed,
            mode="resample",
        )
        map_path = Path(output_dir) / f"{dataset.name}_preference_map.png"
        results.append(
            map_soc_files(
                soc_files,
                map_path,
                max_per_source=map_per_source,
                seed=dataset_seed,
                cultures=("impartial", "mallows"),
                mallows_norm_phi=mallows_norm_phi,
                title=(
                    f"{dataset.name}: approval completions, impartial culture, "
                    f"and Norm-Mallows {mallows_norm_phi:g}"
                ),
            )
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample the French PrefLib approval elections into ordinal profiles "
            "and map them beside IJCAI23 impartial and Mallows cultures."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=Path,
        default=DEFAULT_DATASETS,
        help="Approval dataset directories (default: both French datasets)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "preference_map_outputs",
    )
    parser.add_argument(
        "--num-rankings",
        type=int,
        default=250,
        help="Ordinal rankings sampled from each approval election",
    )
    parser.add_argument(
        "--map-per-source",
        type=int,
        default=75,
        help="Maximum rankings from each source included in an MDS map",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--mallows-norm-phi", type=float, default=0.2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(
        args.datasets,
        args.output_dir,
        num_rankings=args.num_rankings,
        map_per_source=args.map_per_source,
        seed=args.seed,
        mallows_norm_phi=args.mallows_norm_phi,
    )


if __name__ == "__main__":
    main()
