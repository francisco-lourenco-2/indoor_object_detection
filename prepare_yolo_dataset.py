#!/usr/bin/env python3
"""Export indoor detection dataset to Ultralytics YOLO layout (image symlinks + label txts)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from analyse_data import (
    CLASS_NAMES,
    DEFAULT_DATA_ROOT,
    SPLIT_LIST_FILES,
    SPLIT_NAMES,
    Box,
    collect_sequence_images,
    load_all_annotations,
    load_split_image_names,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "prepared_data" / "yolo_indoor_object_detection"
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def boxes_to_yolo_lines(boxes: list[Box], img_w: int, img_h: int) -> list[str]:
    """Convert pixel boxes (top-left + size) to YOLO normalized ``cls cx cy w h`` lines."""
    if img_w <= 0 or img_h <= 0:
        return []
    lines: list[str] = []
    for box in boxes:
        class_id = CLASS_TO_ID.get(box.label)
        if class_id is None:
            print(f"Warning: unknown label {box.label!r}, skipping box.", file=sys.stderr)
            continue
        cx = (box.left + box.width / 2.0) / img_w
        cy = (box.top + box.height / 2.0) / img_h
        w = box.width / img_w
        h = box.height / img_h
        lines.append(
            f"{class_id} {_clip01(cx):.6f} {_clip01(cy):.6f} {_clip01(w):.6f} {_clip01(h):.6f}"
        )
    return lines


def _ensure_symlink(src: Path, dst: Path) -> None:
    src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src:
            return
        dst.unlink()
    dst.symlink_to(src)


def write_indoor_yaml(output_root: Path) -> Path:
    yaml_path = output_root / "indoor.yaml"
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    content = f"""# Ultralytics YOLO dataset — TUT indoor object detection
path: {output_root.resolve()}
train: images/train
val: images/val
test: images/test

nc: {len(CLASS_NAMES)}
names:
{names_block}
"""
    yaml_path.write_text(content)
    return yaml_path


def prepare_yolo_dataset(
    data_root: Path,
    output_root: Path,
    *,
    overwrite_labels: bool = True,
) -> dict[str, int]:
    """
    Build ``images/{{train,val,test}}`` (symlinks) and ``labels/{{train,val,test}}`` (YOLO txt).
    """
    annotations = load_all_annotations(data_root)
    path_by_name = {p.name: p.resolve() for p in collect_sequence_images(data_root)}

    counts: dict[str, int] = {s: 0 for s in SPLIT_NAMES}
    missing_images: list[str] = []

    for split in SPLIT_NAMES:
        try:
            filenames = sorted(load_split_image_names(data_root, split))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{exc}. Run generate_splits.py before preparing YOLO data."
            ) from exc

        img_dir = output_root / "images" / split
        lbl_dir = output_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for filename in filenames:
            src = path_by_name.get(filename)
            if src is None:
                missing_images.append(filename)
                continue

            dst_img = img_dir / filename
            _ensure_symlink(src, dst_img)

            label_path = lbl_dir / f"{Path(filename).stem}.txt"
            boxes = annotations.get(filename, [])
            if boxes:
                with Image.open(src) as im:
                    img_w, img_h = im.size
                lines = boxes_to_yolo_lines(boxes, img_w, img_h)
            else:
                lines = []

            if label_path.exists() and not overwrite_labels:
                pass
            else:
                label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

            counts[split] += 1

    if missing_images:
        print(
            f"Warning: {len(missing_images)} split filenames not found on disk "
            f"(first: {missing_images[0]})",
            file=sys.stderr,
        )

    write_indoor_yaml(output_root)
    return counts


def run_prepare_yolo_dataset(
    data_root: Path | str,
    output_root: Path | str | None = None,
    *,
    overwrite_labels: bool = True,
    verbose: bool = True,
) -> dict[str, int]:
    """Export YOLO layout (symlinks + labels + ``indoor.yaml``). Returns per-split image counts."""
    data_root = Path(data_root).resolve()
    output_root = Path(output_root or DEFAULT_OUTPUT_ROOT).resolve()

    if verbose:
        print(f"Source: {data_root}")
        print(f"Output: {output_root}")
        print(f"Split lists: {', '.join(SPLIT_LIST_FILES[s] for s in SPLIT_NAMES)}")

    counts = prepare_yolo_dataset(data_root, output_root, overwrite_labels=overwrite_labels)

    if verbose:
        print()
        print("=== Export summary ===")
        for split in SPLIT_NAMES:
            print(f"  {split:<5}  {counts[split]:>5} images  ->  images/{split}/  labels/{split}/")
        print(f"  yaml   {output_root / 'indoor.yaml'}")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Ultralytics YOLO dataset layout.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Raw dataset root with sequences and split lists (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"YOLO export directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    args = parser.parse_args()
    run_prepare_yolo_dataset(args.data_root, args.output_root)


if __name__ == "__main__":
    main()
