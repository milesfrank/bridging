"""Sample strict ordinal rankings that are consistent with approval ballots.

For an approval ballot A over candidates C, a consistent strict ranking places
every member of A above every member of C \ A.  A uniform completion is sampled
without enumeration by independently shuffling the approved and disapproved
tiers and concatenating them.

The default ``resample`` mode first draws an approval ballot in proportion to
its PrefLib multiplicity and then draws a uniform completion of that ballot.
The output is an aggregated PrefLib/Mapel ``.soc`` file with zero-based IDs,
matching the IJCAI23 code bundled in this repository.
"""

from __future__ import annotations

import argparse
import bisect
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


HEADER_RE = re.compile(r"^#\s*(.+?):\s*(.*)\s*$")
ALT_NAME_RE = re.compile(r"^ALTERNATIVE NAME\s+(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class ApprovalBallot:
    count: int
    approved: tuple[int, ...]


@dataclass(frozen=True)
class ApprovalElection:
    path: Path
    title: str
    candidate_names: tuple[str, ...]
    ballots: tuple[ApprovalBallot, ...]
    num_voters: int

    @property
    def num_candidates(self) -> int:
        return len(self.candidate_names)


def _split_categories(payload: str) -> list[str]:
    """Split a category list while ignoring commas enclosed in braces."""
    categories: list[str] = []
    current: list[str] = []
    depth = 0
    for char in payload:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Unbalanced category braces in {payload!r}")
        if char == "," and depth == 0:
            categories.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if depth != 0:
        raise ValueError(f"Unbalanced category braces in {payload!r}")
    categories.append("".join(current).strip())
    return categories


def _parse_candidate_set(token: str) -> set[int]:
    token = token.strip()
    if token in {"", "{}"}:
        return set()
    if token.startswith("{") and token.endswith("}"):
        token = token[1:-1].strip()
    return {int(value.strip()) for value in token.split(",") if value.strip()}


def parse_preflib_cat(path: Path) -> ApprovalElection:
    """Parse a complete, two-category PrefLib approval election."""
    path = Path(path)
    num_candidates: Optional[int] = None
    header_voters: Optional[int] = None
    num_categories: Optional[int] = None
    title = path.stem
    names: dict[int, str] = {}
    raw_ballots: list[tuple[int, set[int], set[int]]] = []

    with path.open(encoding="utf-8-sig") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                match = HEADER_RE.match(line)
                if not match:
                    continue
                key, value = match.group(1).strip(), match.group(2).strip()
                upper_key = key.upper()
                if upper_key == "TITLE":
                    title = value or title
                elif upper_key == "NUMBER ALTERNATIVES":
                    num_candidates = int(value)
                elif upper_key == "NUMBER VOTERS":
                    header_voters = int(value)
                elif upper_key == "NUMBER CATEGORIES":
                    num_categories = int(value)
                else:
                    name_match = ALT_NAME_RE.match(key)
                    if name_match:
                        names[int(name_match.group(1))] = value
                continue

            try:
                count_text, payload = line.split(":", 1)
                count = int(count_text.strip())
                categories = _split_categories(payload)
                if len(categories) != 2:
                    raise ValueError(
                        f"expected two approval categories, found {len(categories)}"
                    )
                approved = _parse_candidate_set(categories[0])
                disapproved = _parse_candidate_set(categories[1])
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if count <= 0:
                raise ValueError(f"{path}:{line_number}: ballot count must be positive")
            raw_ballots.append((count, approved, disapproved))

    if num_categories is not None and num_categories != 2:
        raise ValueError(f"{path}: expected 2 categories, found {num_categories}")
    if num_candidates is None:
        observed = set(names)
        for _, approved, disapproved in raw_ballots:
            observed.update(approved)
            observed.update(disapproved)
        if not observed:
            raise ValueError(f"{path}: cannot determine the candidates")
        num_candidates = max(observed)

    all_candidates = set(range(1, num_candidates + 1))
    ballots: list[ApprovalBallot] = []
    for count, approved, disapproved in raw_ballots:
        if approved & disapproved:
            overlap = sorted(approved & disapproved)
            raise ValueError(f"{path}: candidates occur in both categories: {overlap}")
        if approved | disapproved != all_candidates:
            missing = sorted(all_candidates - approved - disapproved)
            extra = sorted((approved | disapproved) - all_candidates)
            raise ValueError(
                f"{path}: incomplete ballot (missing={missing}, out_of_range={extra})"
            )
        # Convert PrefLib's one-based IDs to Mapel's zero-based convention.
        ballots.append(ApprovalBallot(count, tuple(sorted(i - 1 for i in approved))))

    num_voters = sum(ballot.count for ballot in ballots)
    if header_voters is not None and num_voters != header_voters:
        raise ValueError(
            f"{path}: ballot multiplicities total {num_voters}, "
            f"but the header says {header_voters}"
        )
    if not ballots:
        raise ValueError(f"{path}: no ballots found")

    candidate_names = tuple(names.get(i, str(i)) for i in range(1, num_candidates + 1))
    return ApprovalElection(path, title, candidate_names, tuple(ballots), num_voters)


def uniform_completion(
    approved: Sequence[int], num_candidates: int, rng: random.Random
) -> tuple[int, ...]:
    """Draw one uniform strict ranking extending a dichotomous ballot."""
    approved_list = list(approved)
    approved_set = set(approved_list)
    disapproved_list = [
        candidate for candidate in range(num_candidates) if candidate not in approved_set
    ]
    rng.shuffle(approved_list)
    rng.shuffle(disapproved_list)
    return tuple(approved_list + disapproved_list)


def sample_rankings(
    election: ApprovalElection,
    num_rankings: Optional[int] = None,
    seed: int = 23,
    mode: str = "resample",
) -> list[tuple[int, ...]]:
    """Sample rankings from an approval election.

    ``resample`` draws approval ballot types according to voter multiplicity;
    ``complete`` draws one independent completion for every original voter.
    """
    if mode not in {"resample", "complete"}:
        raise ValueError("mode must be 'resample' or 'complete'")
    if mode == "complete":
        if num_rankings is not None and num_rankings != election.num_voters:
            raise ValueError(
                "complete mode produces exactly one ranking per original voter "
                f"({election.num_voters})"
            )
        num_rankings = election.num_voters
    elif num_rankings is None:
        num_rankings = election.num_voters
    if num_rankings <= 0:
        raise ValueError("num_rankings must be positive")

    rng = random.Random(seed)
    rankings: list[tuple[int, ...]] = []
    if mode == "complete":
        for ballot in election.ballots:
            for _ in range(ballot.count):
                rankings.append(
                    uniform_completion(ballot.approved, election.num_candidates, rng)
                )
        rng.shuffle(rankings)
        return rankings

    cumulative_counts: list[int] = []
    running_total = 0
    for ballot in election.ballots:
        running_total += ballot.count
        cumulative_counts.append(running_total)
    for _ in range(num_rankings):
        draw = rng.randrange(election.num_voters)
        ballot_index = bisect.bisect_right(cumulative_counts, draw)
        rankings.append(
            uniform_completion(
                election.ballots[ballot_index].approved, election.num_candidates, rng
            )
        )
    return rankings


def write_soc(
    output_path: Path,
    election: ApprovalElection,
    rankings: Iterable[Sequence[int]],
    seed: int,
    mode: str,
) -> None:
    """Write rankings as an aggregated strict-order ``.soc`` file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(tuple(ranking) for ranking in rankings)
    num_rankings = sum(counts.values())
    if num_rankings == 0:
        raise ValueError("cannot write an empty ordinal election")

    metadata = {
        "source": election.path.name,
        "title": election.title,
        "sampling": mode,
        "seed": seed,
        "aggregated": True,
    }
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(f"# approval_completion {metadata!r}\n")
        output.write(f"{election.num_candidates}\n")
        for candidate_id, name in enumerate(election.candidate_names):
            clean_name = name.replace("\r", " ").replace("\n", " ")
            output.write(f"{candidate_id}, {clean_name}\n")
        output.write(f"{num_rankings}, {num_rankings}, {len(counts)}\n")
        for ranking, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            output.write(f"{count}, " + ", ".join(map(str, ranking)) + "\n")


def _expand_inputs(inputs: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for input_path in inputs:
        input_path = Path(input_path)
        if input_path.is_dir():
            files.extend(sorted(input_path.glob("*.cat")))
        elif input_path.suffix.lower() == ".cat" and input_path.is_file():
            files.append(input_path)
        else:
            raise FileNotFoundError(f"not a .cat file or directory: {input_path}")
    if not files:
        raise FileNotFoundError("no .cat files found")
    return files


def sample_files(
    inputs: Sequence[Path],
    output_dir: Path,
    num_rankings: Optional[int] = None,
    seed: int = 23,
    mode: str = "resample",
) -> list[Path]:
    """Sample every input file and return the paths of the written SOC files."""
    input_files = _expand_inputs(inputs)
    written: list[Path] = []
    for file_index, input_file in enumerate(input_files):
        election = parse_preflib_cat(input_file)
        file_seed = seed + file_index
        rankings = sample_rankings(election, num_rankings, file_seed, mode)
        output_path = Path(output_dir) / f"{input_file.stem}.soc"
        write_soc(output_path, election, rankings, file_seed, mode)
        written.append(output_path)
        print(
            f"{input_file.name}: sampled {len(rankings)} rankings "
            f"over {election.num_candidates} candidates -> {output_path}"
        )
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Uniformly complete two-tier PrefLib approval ballots and write "
            "an ordinal .soc election."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more PrefLib .cat files or directories containing them",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preference_map_outputs") / "ordinal_samples",
        help="Directory for sampled .soc files",
    )
    parser.add_argument(
        "-n",
        "--num-rankings",
        type=int,
        default=None,
        help="Number of rankings per input (default: original voter count)",
    )
    parser.add_argument("--seed", type=int, default=23, help="Base random seed")
    parser.add_argument(
        "--mode",
        choices=("resample", "complete"),
        default="resample",
        help=(
            "resample approval ballots by multiplicity, or complete each original "
            "voter exactly once"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sample_files(
        args.inputs,
        args.output_dir,
        num_rankings=args.num_rankings,
        seed=args.seed,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
