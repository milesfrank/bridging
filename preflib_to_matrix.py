"""Convert PrefLib categorical approval files into voter-by-candidate 0/1 matrices.

Each row is a voter, each column is a candidate, and an entry is 1 if the voter
approved that candidate and 0 otherwise. PrefLib .cat files store unique ballots
with a multiplicity count; this script expands those into one row per voter.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEADER_RE = re.compile(r"^#\s*(.+?):\s*(.*)\s*$")
ALT_NAME_RE = re.compile(r"^ALTERNATIVE NAME\s+(\d+)$", re.IGNORECASE)


@dataclass
class ApprovalElection:
    path: Path
    title: str
    candidate_names: list[str]
    matrix: np.ndarray  # shape (n_voters, n_candidates), dtype int


def _split_categories(payload: str) -> list[str]:
    """Split Yes/No category strings, ignoring commas inside braces."""
    categories: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in payload:
        if ch == "{":
            depth += 1
            current.append(ch)
        elif ch == "}":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            categories.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        categories.append("".join(current).strip())
    return categories


def _parse_alt_set(token: str) -> set[int]:
    token = token.strip()
    if not token or token == "{}":
        return set()
    if token.startswith("{") and token.endswith("}"):
        inner = token[1:-1].strip()
        if not inner:
            return set()
        return {int(x.strip()) for x in inner.split(",") if x.strip()}
    return {int(token)}


def parse_preflib_cat(path: Path) -> ApprovalElection:
    n_alts: int | None = None
    n_voters_header: int | None = None
    title = path.stem
    names: dict[int, str] = {}
    ballots: list[tuple[int, set[int]]] = []

    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                match = HEADER_RE.match(line)
                if not match:
                    continue
                key, value = match.group(1).strip(), match.group(2).strip()
                if key.upper() == "TITLE":
                    title = value
                elif key.upper() == "NUMBER ALTERNATIVES":
                    n_alts = int(value)
                elif key.upper() == "NUMBER VOTERS":
                    n_voters_header = int(value)
                else:
                    alt_match = ALT_NAME_RE.match(key)
                    if alt_match:
                        names[int(alt_match.group(1))] = value
                continue

            count_str, payload = line.split(":", 1)
            count = int(count_str.strip())
            categories = _split_categories(payload)
            approved = _parse_alt_set(categories[0]) if categories else set()
            ballots.append((count, approved))

    if n_alts is None:
        n_alts = max(names) if names else max((max(s) for _, s in ballots if s), default=0)

    candidate_names = [names.get(i, str(i)) for i in range(1, n_alts + 1)]
    n_voters = sum(count for count, _ in ballots)
    matrix = np.zeros((n_voters, n_alts), dtype=int)

    row = 0
    for count, approved in ballots:
        row_vec = np.zeros(n_alts, dtype=int)
        for alt in approved:
            row_vec[alt - 1] = 1
        matrix[row : row + count] = row_vec
        row += count

    if n_voters_header is not None and n_voters != n_voters_header:
        raise ValueError(
            f"{path.name}: expanded {n_voters} voters, header says {n_voters_header}"
        )

    return ApprovalElection(path, title, candidate_names, matrix)


def load_directory(data_dir: Path) -> list[ApprovalElection]:
    cat_files = sorted(data_dir.glob("*.cat"))
    if not cat_files:
        raise FileNotFoundError(f"No .cat files found in {data_dir}")
    return [parse_preflib_cat(p) for p in cat_files]


def combine_elections(elections: list[ApprovalElection]) -> ApprovalElection:
    if not elections:
        raise ValueError("No elections to combine")
    names = elections[0].candidate_names
    for el in elections[1:]:
        if el.candidate_names != names:
            raise ValueError("Candidate names/order differ across files")
    matrix = np.vstack([el.matrix for el in elections])
    return ApprovalElection(
        path=elections[0].path.parent,
        title="combined",
        candidate_names=names,
        matrix=matrix,
    )


def write_csv(election: ApprovalElection, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(election.candidate_names)
        writer.writerows(election.matrix.tolist())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Turn PrefLib French approval .cat files into 0/1 matrices."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "00026_frenchapproval",
        help="Directory containing PrefLib .cat files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="If set, write one CSV per election (header = candidate names)",
    )
    parser.add_argument(
        "--combined",
        type=Path,
        default=None,
        help="If set, write a single stacked CSV of all voters",
    )
    args = parser.parse_args()

    elections = load_directory(args.data_dir)
    for el in elections:
        n, m = el.matrix.shape
        print(f"{el.path.name} ({el.title}): {n} voters x {m} candidates")
        print("  candidates:", ", ".join(el.candidate_names))
        print(f"  approvals per voter: min={el.matrix.sum(axis=1).min()}, "
              f"max={el.matrix.sum(axis=1).max()}, "
              f"mean={el.matrix.sum(axis=1).mean():.2f}")
        if args.out_dir:
            out_path = args.out_dir / f"{el.path.stem}.csv"
            write_csv(el, out_path)
            print(f"  wrote {out_path}")

    if args.combined:
        combined = combine_elections(elections)
        write_csv(combined, args.combined)
        n, m = combined.matrix.shape
        print(f"combined: {n} voters x {m} candidates")
        print(f"  wrote {args.combined}")


if __name__ == "__main__":
    main()
