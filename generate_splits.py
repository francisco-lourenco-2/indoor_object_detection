#!/usr/bin/env python3
"""Build stratified train/val/test splits via integer linear programming (PuLP)."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pulp

from analyse_data import (
    CLASS_NAMES,
    DEFAULT_DATA_ROOT,
    collect_sequence_images,
    load_all_annotations,
)

# Class columns: chair, clock, exit, fireextinguisher, printer, screen, trashbin
N_CLASSES = len(CLASS_NAMES)
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# 80 / 10 / 10 split (ILP targets will use these ratios).
SPLIT_RATIOS: dict[str, float] = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}
SPLIT_NAMES: tuple[str, ...] = tuple(SPLIT_RATIOS.keys())
# Hard equality on instance counts; all other classes are soft in the objective.
HARD_CLASS_NAMES: tuple[str, ...] = ("printer", "screen")
HARD_CLASS_INDICES: tuple[int, ...] = tuple(CLASS_TO_INDEX[n] for n in HARD_CLASS_NAMES)
SOFT_CLASS_INDICES: tuple[int, ...] = tuple(
    i for i in range(N_CLASSES) if i not in HARD_CLASS_INDICES
)
# val split is written to valid.txt
SPLIT_TO_LIST_FILE: dict[str, str] = {
    "train": "train.txt",
    "val": "valid.txt",
    "test": "test.txt",
}


def build_occurrence_data(
    images: list[Path],
    annotations: dict,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Build ``occurrence_array`` (n_images × n_classes) and ``image_dict`` (filename -> row index).

    Rows follow sorted image filenames so order is stable across runs.

    Example::

        occurrence_array[image_dict["frame_s4_31.jpg"], 4]  # printer count in that frame
    """
    filenames = sorted(p.name for p in images)
    n_images = len(filenames)
    occurrence_array = np.zeros((n_images, N_CLASSES), dtype=np.int32)
    image_dict: dict[str, int] = {name: i for i, name in enumerate(filenames)}

    for path in images:
        row = image_dict[path.name]
        for box in annotations.get(path.name, []):
            col = CLASS_TO_INDEX.get(box.label)
            if col is not None:
                occurrence_array[row, col] += 1

    return occurrence_array, image_dict


def class_occurrence_totals(occurrence_array: np.ndarray) -> np.ndarray:
    """Per-class instance counts over the full dataset (sum over image rows)."""
    return occurrence_array.sum(axis=0)


def allocate_integer_split_targets(
    totals: np.ndarray,
    split_ratios: dict[str, float],
) -> dict[str, np.ndarray]:
    """
    Integer targets per class per split: proportional to ``total × ratio``, summing to ``totals``.

    Uses largest-remainder allocation per class so train+val+test equals the dataset total exactly.
    """
    split_names = list(split_ratios.keys())
    n_classes = len(totals)
    out = {name: np.zeros(n_classes, dtype=np.int64) for name in split_names}

    for c in range(n_classes):
        total = int(totals[c])
        if total == 0:
            continue
        quotas = np.array([total * split_ratios[name] for name in split_names], dtype=np.float64)
        base = np.floor(quotas).astype(np.int64)
        remainder = total - int(base.sum())
        if remainder > 0:
            order = np.argsort(quotas - base)[::-1]
            for k in range(remainder):
                base[order[k]] += 1
        for i, name in enumerate(split_names):
            out[name][c] = base[i]

    return out


def compute_split_targets(
    occurrence_array: np.ndarray,
    split_ratios: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """
    Integer target instance count per class per split (proportional to dataset totals × ratio).
    """
    ratios = split_ratios if split_ratios is not None else SPLIT_RATIOS
    totals = class_occurrence_totals(occurrence_array)
    return allocate_integer_split_targets(totals, ratios)


def print_split_targets(
    occurrence_array: np.ndarray,
    split_ratios: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Print dataset-wide class totals and per-split target occurrence counts."""
    ratios = split_ratios if split_ratios is not None else SPLIT_RATIOS
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        print(f"Warning: split ratios sum to {sum(ratios.values()):.6f}, not 1.0.", file=sys.stderr)

    totals = class_occurrence_totals(occurrence_array)
    targets = compute_split_targets(occurrence_array, ratios)
    n_images = occurrence_array.shape[0]
    total_boxes = int(totals.sum())

    print()
    print("=== Dataset totals (all images) ===")
    print(f"Images: {n_images}   Total bounding boxes: {total_boxes}")
    print(f"{'class':<18} {'instances':>12}")
    print("-" * 32)
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<18} {int(totals[i]):>12}")

    print()
    print("=== Target instance counts per split (integer, proportional to ratio) ===")
    print(f"Ratios: {', '.join(f'{k}={v:.0%}' for k, v in ratios.items())}")
    header = f"{'class':<18} {'dataset':>10}"
    for split in ratios:
        header += f" {split:>10}"
    print(header)
    print("-" * len(header))
    for i, name in enumerate(CLASS_NAMES):
        row = f"{name:<18} {int(totals[i]):>10}"
        for split in ratios:
            row += f" {int(targets[split][i]):>10}"
        print(row)

    print()
    print("=== Target totals per split (sum over classes) ===")
    for split in ratios:
        split_total = int(targets[split].sum())
        print(f"  {split:<6} {split_total:>10}  (of {total_boxes} total boxes)")

    return targets


def allocate_integer_image_counts(
    n_images: int,
    split_ratios: dict[str, float] | None = None,
) -> dict[str, int]:
    """80/10/10 (or custom) integer image counts summing to ``n_images``."""
    ratios = split_ratios if split_ratios is not None else SPLIT_RATIOS
    per_split = allocate_integer_split_targets(np.array([n_images], dtype=np.int64), ratios)
    return {name: int(per_split[name][0]) for name in ratios}


def _filenames_in_row_order(image_dict: dict[str, int]) -> list[str]:
    n = len(image_dict)
    names = [""] * n
    for filename, idx in image_dict.items():
        names[idx] = filename
    return names


def _soft_class_weights(totals: np.ndarray) -> dict[int, float]:
    """Weight soft deviations inversely to dataset-wide class frequency."""
    weights = {c: 1.0 / max(int(totals[c]), 1) for c in SOFT_CLASS_INDICES}
    total_w = sum(weights.values())
    return {c: w / total_w for c, w in weights.items()}


@dataclass
class SplitSolution:
    assignments: dict[str, list[str]]
    status: str
    objective: float | None


def solve_split_ilp(
    occurrence_array: np.ndarray,
    image_dict: dict[str, int],
    targets: dict[str, np.ndarray],
    split_ratios: dict[str, float] | None = None,
    *,
    time_limit_sec: int = 600,
    msg: bool = False,
) -> SplitSolution:
    """
    ILP: assign each image to exactly one split.

    - Hard: printer & screen instance counts match integer targets per split.
    - Hard: image counts per split follow the split ratios (integer allocation).
    - Hard: every class has at least one instance in every split.
    - Soft: minimize weighted L1 deviation from targets (other classes);
      weights are inversely proportional to dataset class totals.
    """
    ratios = split_ratios if split_ratios is not None else SPLIT_RATIOS
    splits = list(ratios.keys())
    n_images, _ = occurrence_array.shape
    totals = class_occurrence_totals(occurrence_array)
    weights = _soft_class_weights(totals)
    image_targets = allocate_integer_image_counts(n_images, ratios)
    filenames = _filenames_in_row_order(image_dict)

    prob = pulp.LpProblem("docusketch_split", pulp.LpMinimize)
    x = pulp.LpVariable.dicts(
        "x",
        ((i, s) for i in range(n_images) for s in splits),
        cat=pulp.LpBinary,
    )

    # Each image in exactly one split.
    for i in range(n_images):
        prob += pulp.lpSum(x[i, s] for s in splits) == 1, f"one_split_{i}"

    # Target number of images per split (80/10/10).
    for s in splits:
        prob += (
            pulp.lpSum(x[i, s] for i in range(n_images)) == image_targets[s],
            f"num_images_{s}",
        )

    # Hard class instance counts (printer, screen).
    for s in splits:
        for c in HARD_CLASS_INDICES:
            prob += (
                pulp.lpSum(int(occurrence_array[i, c]) * x[i, s] for i in range(n_images))
                == int(targets[s][c]),
                f"hard_{s}_{CLASS_NAMES[c]}",
            )

    # Every class represented in every split (>= 1 instance).
    for s in splits:
        for c in range(N_CLASSES):
            if int(totals[c]) == 0:
                continue
            prob += (
                pulp.lpSum(int(occurrence_array[i, c]) * x[i, s] for i in range(n_images)) >= 1,
                f"present_{s}_{CLASS_NAMES[c]}",
            )

    # Soft deviations for non-hard classes.
    deviation = pulp.LpVariable.dicts(
        "dev",
        ((s, c) for s in splits for c in SOFT_CLASS_INDICES),
        lowBound=0,
    )
    for s in splits:
        for c in SOFT_CLASS_INDICES:
            actual = pulp.lpSum(int(occurrence_array[i, c]) * x[i, s] for i in range(n_images))
            tgt = int(targets[s][c])
            prob += deviation[s, c] >= actual - tgt, f"dev_pos_{s}_{c}"
            prob += deviation[s, c] >= tgt - actual, f"dev_neg_{s}_{c}"

    prob += pulp.lpSum(
        weights[c] * deviation[s, c] for s in splits for c in SOFT_CLASS_INDICES
    ), "weighted_l1_deviation"

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_sec)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    objective = pulp.value(prob.objective) if prob.status == pulp.LpStatusOptimal else None

    assignments: dict[str, list[str]] = {s: [] for s in splits}
    if status != "Optimal":
        return SplitSolution(assignments=assignments, status=status, objective=objective)

    for i in range(n_images):
        for s in splits:
            if pulp.value(x[i, s]) > 0.5:
                assignments[s].append(filenames[i])
                break

    for s in splits:
        assignments[s].sort()

    return SplitSolution(assignments=assignments, status=status, objective=objective)


def achieved_class_counts(
    occurrence_array: np.ndarray,
    image_dict: dict[str, int],
    assignments: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    """Per-split achieved instance counts per class."""
    out = {s: np.zeros(N_CLASSES, dtype=np.int64) for s in assignments}
    for split, names in assignments.items():
        for name in names:
            row = image_dict[name]
            out[split] += occurrence_array[row]
    return out


def print_solution_distributions(
    targets: dict[str, np.ndarray],
    achieved: dict[str, np.ndarray],
    assignments: dict[str, list[str]],
    *,
    objective: float | None = None,
    status: str = "Optimal",
) -> None:
    """Compare target vs achieved instance counts per class per split."""
    print()
    print(f"=== ILP solution ({status}) ===")
    if objective is not None:
        print(f"Weighted soft objective: {objective:.6f}")

    splits = list(targets.keys())
    header = f"{'class':<18} {'constraint':>10}"
    for s in splits:
        header += f" {s + ' tgt':>10} {s + ' got':>10} {s + ' Δ':>8}"
    print(header)
    print("-" * len(header))

    for i, name in enumerate(CLASS_NAMES):
        kind = "hard" if i in HARD_CLASS_INDICES else "soft"
        row = f"{name:<18} {kind:>10}"
        for s in splits:
            tgt = int(targets[s][i])
            got = int(achieved[s][i])
            delta = got - tgt
            row += f" {tgt:>10} {got:>10} {delta:>8}"
        print(row)

    print()
    print("=== Per-split totals (instances) ===")
    for s in splits:
        print(
            f"  {s:<6}  target {int(targets[s].sum()):>5}   achieved {int(achieved[s].sum()):>5}   "
            f"images {len(assignments[s])}"
        )


def write_split_list_files(
    assignments: dict[str, list[str]],
    data_root: Path,
) -> None:
    """Write train.txt, valid.txt, test.txt under ``data_root``."""
    for split, filenames in assignments.items():
        list_name = SPLIT_TO_LIST_FILE[split]
        path = data_root / list_name
        path.write_text("\n".join(filenames) + ("\n" if filenames else ""))
        print(f"Wrote {len(filenames)} filenames to {path}")


def _interactive_dump(occurrence_array: np.ndarray, image_dict: dict[str, int]) -> None:
    """TEMP: print one row per Enter key."""
    filenames = sorted(image_dict, key=image_dict.get)
    print(
        f"occurrence_array shape {occurrence_array.shape} "
        f"({occurrence_array.size} = n_images × n_classes). Press Enter after each line.\n"
    )
    for filename in filenames:
        row = image_dict[filename]
        print(f"{filename}: {occurrence_array[row]}")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            print("\nStopped.")
            break


@dataclass
class SplitRunResult:
    solution: SplitSolution
    targets: dict[str, np.ndarray]
    achieved: dict[str, np.ndarray]
    occurrence_array: np.ndarray
    image_dict: dict[str, int]


def run_generate_splits(
    data_root: Path | str,
    *,
    time_limit_sec: int = 600,
    solver_msg: bool = False,
    targets_only: bool = False,
    verbose: bool = True,
) -> SplitRunResult | None:
    """
    Build occurrence matrix, solve the split ILP, and write ``train.txt`` / ``valid.txt`` / ``test.txt``.

    Returns None if ``targets_only=True`` (targets printed only).
    """
    data_root = Path(data_root).resolve()
    images = collect_sequence_images(data_root)
    annotations = load_all_annotations(data_root)
    occurrence_array, image_dict = build_occurrence_data(images, annotations)

    if verbose:
        print(f"Built occurrence_array {occurrence_array.shape} and image_dict ({len(image_dict)} images).")
        print(f"Class column order: {list(CLASS_NAMES)}")
        print(f"Hard classes (equality): {list(HARD_CLASS_NAMES)}")
        soft_weights = _soft_class_weights(class_occurrence_totals(occurrence_array))
        print("Soft class weights (normalized inverse frequency):")
        for c in SOFT_CLASS_INDICES:
            print(f"  {CLASS_NAMES[c]:<18} {soft_weights[c]:.6f}")

    targets = print_split_targets(occurrence_array, SPLIT_RATIOS)

    if targets_only:
        return None

    image_counts = allocate_integer_image_counts(occurrence_array.shape[0], SPLIT_RATIOS)
    if verbose:
        print()
        print("=== Target image counts per split ===")
        for s in SPLIT_NAMES:
            print(f"  {s:<6} {image_counts[s]}")

    solution = solve_split_ilp(
        occurrence_array,
        image_dict,
        targets,
        time_limit_sec=time_limit_sec,
        msg=solver_msg,
    )

    if solution.status != "Optimal":
        raise RuntimeError(f"ILP failed with status: {solution.status}")

    achieved = achieved_class_counts(occurrence_array, image_dict, solution.assignments)
    print_solution_distributions(
        targets,
        achieved,
        solution.assignments,
        objective=solution.objective,
        status=solution.status,
    )

    write_split_list_files(solution.assignments, data_root)

    return SplitRunResult(
        solution=solution,
        targets=targets,
        achieved=achieved,
        occurrence_array=occurrence_array,
        image_dict=image_dict,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-image class occurrence matrix.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Dataset root (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Print each image row and wait for Enter (debug)",
    )
    parser.add_argument(
        "--targets-only",
        action="store_true",
        help="Only print target distributions (skip ILP)",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=600,
        help="CBC time limit in seconds (default: 600)",
    )
    parser.add_argument(
        "--solver-msg",
        action="store_true",
        help="Print CBC solver log",
    )
    args = parser.parse_args()
    data_root = args.data_root.resolve()

    try:
        result = run_generate_splits(
            data_root,
            time_limit_sec=args.time_limit,
            solver_msg=args.solver_msg,
            targets_only=args.targets_only,
            verbose=True,
        )
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    if args.interactive and result is not None:
        _interactive_dump(result.occurrence_array, result.image_dict)


if __name__ == "__main__":
    main()
