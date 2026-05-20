#!/usr/bin/env python3
"""Dataset analysis: print statistics (default) or browse images with overlays (--visualize)."""

from __future__ import annotations

import argparse
import random
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
DATASET_SUBDIR = "indoor_object_detection_dataset"
DEFAULT_DATA_ROOT = DATA_DIR / DATASET_SUBDIR
ANNOTATION_DIR_NAME = "annotation"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
WINDOW_NAME = "DocuSketch — image viewer"

LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "chair": (255, 128, 0),
    "clock": (0, 200, 255),
    "exit": (0, 255, 0),
    "fireextinguisher": (0, 0, 255),
    "printer": (255, 0, 255),
    "screen": (255, 255, 0),
    "trashbin": (180, 105, 255),
}
CLASS_NAMES: tuple[str, ...] = tuple(LABEL_COLORS.keys())
_DEFAULT_BOX_COLOR = (200, 200, 200)

# Split lists written by generate_splits.py (val -> valid.txt).
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
SPLIT_LIST_FILES: dict[str, str] = {
    "train": "train.txt",
    "val": "valid.txt",
    "test": "test.txt",
}

_ARROW_NEXT = {65363, 2555904}
_ARROW_PREV = {65361, 2424832}
_ARROW_NEXT_LOW = {83, 3}
_ARROW_PREV_LOW = {81, 2}


@dataclass(frozen=True)
class Box:
    label: str
    left: int
    top: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def x2(self) -> int:
        return self.left + self.width

    @property
    def y2(self) -> int:
        return self.top + self.height


@dataclass
class ClassStats:
    label: str
    occurrences: int
    images_with_object: int
    avg_bbox_area: float
    avg_occurrences_per_image: float


@dataclass
class DatasetStats:
    total_images: int
    images_with_objects: int
    images_without_objects: int
    annotated_frames_in_xml: int
    images_missing_xml: int
    xml_entries_without_image: int
    total_boxes: int
    per_class: list[ClassStats]


def _navigation_step(key: int) -> int | None:
    low = key & 0xFF
    if key in _ARROW_NEXT or low in _ARROW_NEXT_LOW:
        return 1
    if key in _ARROW_PREV or low in _ARROW_PREV_LOW:
        return -1
    return None


def _parse_box_element(box_el: ET.Element) -> Box | None:
    try:
        left = int(box_el.get("left", ""))
        top = int(box_el.get("top", ""))
        width = int(box_el.get("width", ""))
        height = int(box_el.get("height", ""))
    except (TypeError, ValueError):
        return None
    label_el = box_el.find("label")
    label = (label_el.text or "").strip() if label_el is not None else ""
    if not label:
        label = "?"
    return Box(label=label, left=left, top=top, width=width, height=height)


def parse_annotation_xml(xml_path: Path) -> dict[str, list[Box]]:
    root = ET.parse(xml_path).getroot()
    out: dict[str, list[Box]] = {}
    for image_el in root.findall(".//image"):
        filename = image_el.get("file")
        if not filename:
            continue
        boxes: list[Box] = []
        for box_el in image_el.findall("box"):
            box = _parse_box_element(box_el)
            if box is not None:
                boxes.append(box)
        out[filename] = boxes
    return out


def load_all_annotations(data_root: Path) -> dict[str, list[Box]]:
    ann_dir = data_root / ANNOTATION_DIR_NAME
    merged: dict[str, list[Box]] = {}
    if not ann_dir.is_dir():
        print(f"Warning: no {ann_dir}.", file=sys.stderr)
        return merged
    for xml_path in sorted(ann_dir.glob("annotation_s*.xml")):
        for filename, boxes in parse_annotation_xml(xml_path).items():
            merged[filename] = boxes
    return merged


def image_contains_class(
    image_name: str,
    annotations: dict[str, list[Box]],
    class_name: str,
) -> bool:
    return any(box.label == class_name for box in annotations.get(image_name, []))


def filter_images_by_class(
    images: list[Path],
    annotations: dict[str, list[Box]],
    class_name: str,
) -> list[Path]:
    return [p for p in images if image_contains_class(p.name, annotations, class_name)]


def load_split_image_names(data_root: Path, split: str) -> set[str]:
    """Filenames listed in train.txt / valid.txt / test.txt under ``data_root``."""
    if split not in SPLIT_LIST_FILES:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLIT_NAMES}")
    list_path = data_root / SPLIT_LIST_FILES[split]
    if not list_path.is_file():
        raise FileNotFoundError(
            f"Split list not found: {list_path} (run generate_splits.py first)"
        )
    return {
        line.strip()
        for line in list_path.read_text().splitlines()
        if line.strip()
    }


def filter_images_by_split(images: list[Path], split_names: set[str]) -> list[Path]:
    return [p for p in images if p.name in split_names]


def collect_sequence_images(data_root: Path) -> list[Path]:
    paths: list[Path] = []
    for seq_dir in sorted(data_root.glob("sequence_*")):
        if not seq_dir.is_dir():
            continue
        for p in seq_dir.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(p)
    return sorted(paths)


def compute_dataset_stats(
    images: list[Path],
    annotations: dict[str, list[Box]],
) -> DatasetStats:
    image_names = {p.name for p in images}
    xml_names = set(annotations)

    occurrences: dict[str, int] = defaultdict(int)
    area_sum: dict[str, float] = defaultdict(float)
    images_with_label: dict[str, set[str]] = defaultdict(set)
    per_image_counts: dict[str, list[int]] = defaultdict(list)

    images_with_objects = 0
    total_boxes = 0

    for path in images:
        boxes = annotations.get(path.name, [])
        total_boxes += len(boxes)
        if boxes:
            images_with_objects += 1
        counts_this_image: dict[str, int] = defaultdict(int)
        for box in boxes:
            label = box.label
            occurrences[label] += 1
            area_sum[label] += box.area
            counts_this_image[label] += 1
        for label, count in counts_this_image.items():
            images_with_label[label].add(path.name)
            per_image_counts[label].append(count)

    per_class: list[ClassStats] = []
    for label in sorted(occurrences):
        n_img = len(images_with_label[label])
        n_occ = occurrences[label]
        per_class.append(
            ClassStats(
                label=label,
                occurrences=n_occ,
                images_with_object=n_img,
                avg_bbox_area=area_sum[label] / n_occ,
                avg_occurrences_per_image=n_occ / n_img if n_img else 0.0,
            )
        )

    return DatasetStats(
        total_images=len(images),
        images_with_objects=images_with_objects,
        images_without_objects=len(images) - images_with_objects,
        annotated_frames_in_xml=len(annotations),
        images_missing_xml=sum(1 for p in images if p.name not in annotations),
        xml_entries_without_image=len(xml_names - image_names),
        total_boxes=total_boxes,
        per_class=per_class,
    )


def print_dataset_stats(stats: DatasetStats, data_root: Path) -> None:
    print(f"Dataset: {data_root}")
    print()
    print("=== Overview ===")
    print(f"Total images:                         {stats.total_images}")
    print(f"Images with annotated objects:        {stats.images_with_objects}")
    print(f"Images without annotated objects:     {stats.images_without_objects}")
    print(f"Total bounding boxes:                 {stats.total_boxes}")
    if stats.images_missing_xml or stats.xml_entries_without_image:
        print()
        print("=== Annotation / image alignment ===")
        print(f"Images missing XML entry:             {stats.images_missing_xml}")
        print(f"XML entries without image on disk:    {stats.xml_entries_without_image}")
        print(f"Annotated frames in XML (total):    {stats.annotated_frames_in_xml}")

    print()
    print("=== Per object class ===")
    header = (
        f"{'class':<18} {'occurrences':>12} {'images':>10} "
        f"{'avg bbox area':>14} {'avg per image':>14}"
    )
    print(header)
    print("-" * len(header))
    for row in stats.per_class:
        print(
            f"{row.label:<18} {row.occurrences:>12} {row.images_with_object:>10} "
            f"{row.avg_bbox_area:>14.1f} {row.avg_occurrences_per_image:>14.2f}"
        )
    print()
    print(
        "avg per image = mean box count on images that contain at least one "
        "instance of that class"
    )


def draw_annotations(bgr, boxes: list[Box]):
    import cv2

    if not boxes:
        return bgr
    out = bgr.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2

    for box in boxes:
        color = LABEL_COLORS.get(box.label, _DEFAULT_BOX_COLOR)
        x1 = max(0, box.left)
        y1 = max(0, box.top)
        x2 = min(w - 1, box.x2)
        y2 = min(h - 1, box.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        label = box.label
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
        pad = 2
        bar_h = th + baseline + 2 * pad
        bar_y1 = y1 - bar_h if y1 - bar_h >= 0 else y1
        bar_y2 = bar_y1 + bar_h
        bar_x2 = min(w - 1, x1 + tw + 2 * pad)
        cv2.rectangle(out, (x1, bar_y1), (bar_x2, bar_y2), color, -1)
        text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
        cv2.putText(
            out,
            label,
            (x1 + pad, bar_y2 - baseline - pad),
            font,
            font_scale,
            text_color,
            1,
            cv2.LINE_AA,
        )
    return out


def _draw_status_bar(image, text: str, bar_h: int = 28):
    import cv2
    import numpy as np

    h, w = image.shape[:2]
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    bar[:] = (40, 40, 40)
    cv2.putText(
        bar,
        text,
        (8, bar_h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([image, bar])


def browse(
    images: list[Path],
    data_root: Path,
    annotations: dict[str, list[Box]],
    seed: int | None,
    *,
    show_overlays: bool = True,
    filter_class: str | None = None,
    filter_split: str | None = None,
) -> None:
    import cv2

    if not images:
        if filter_class and filter_split:
            print(
                f"No images in split '{filter_split}' contain class '{filter_class}'.",
                file=sys.stderr,
            )
        elif filter_class:
            print(f"No images contain class '{filter_class}'.", file=sys.stderr)
        elif filter_split:
            print(f"No images in split '{filter_split}'.", file=sys.stderr)
        else:
            print(f"No images found under {data_root}/sequence_*", file=sys.stderr)
        sys.exit(1)

    order = images.copy()
    rng = random.Random(seed)
    rng.shuffle(order)

    missing = sum(1 for p in order if p.name not in annotations)
    if show_overlays and missing:
        print(f"Note: {missing} images have no entry in annotation XML.", file=sys.stderr)

    idx = 0
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        path = order[idx]
        bgr = cv2.imread(str(path))
        if bgr is None:
            print(f"Warning: could not read {path}", file=sys.stderr)
            idx = (idx + 1) % len(order)
            if idx == 0:
                break
            continue

        boxes = annotations.get(path.name, []) if show_overlays else []
        if show_overlays and filter_class is not None:
            boxes = [b for b in boxes if b.label == filter_class]
        vis = draw_annotations(bgr, boxes) if boxes else bgr

        try:
            rel = path.relative_to(data_root)
        except ValueError:
            rel = path.name
        nbox = len(boxes)
        overlay_hint = f"{nbox} box{'es' if nbox != 1 else ''}  |  " if show_overlays else ""
        split_hint = f"split={filter_split}  |  " if filter_split else ""
        filter_hint = f"class={filter_class}  |  " if filter_class else ""
        status = (
            f"[{idx + 1}/{len(order)}]  {rel}  |  {split_hint}{filter_hint}{overlay_hint}"
            "← → scroll  |  q: quit"
        )
        display = _draw_status_bar(vis, status)
        cv2.imshow(WINDOW_NAME, display)
        try:
            cv2.setWindowTitle(WINDOW_NAME, str(rel))
        except cv2.error:
            pass

        key = cv2.waitKey(0)
        if key == -1:
            continue
        if (key & 0xFF) in (ord("q"), 27):
            break
        step = _navigation_step(key)
        if step is not None:
            idx = (idx + step) % len(order)

    cv2.destroyAllWindows()


def remove_dataset_zips(data_dir: Path | str) -> list[Path]:
    """Delete dataset archive(s) under ``data_dir`` (any ``*.zip``). Returns removed paths."""
    data_dir = Path(data_dir).resolve()
    removed: list[Path] = []
    for z in sorted(data_dir.glob("*.zip")):
        z.unlink()
        removed.append(z)
        print(f"Removed {z}")
    return removed


def download_indoor_dataset(
    data_dir: Path | str | None = None,
    *,
    force: bool = False,
    remove_zip: bool = True,
) -> Path:
    """
    Download and extract the TUT indoor object detection dataset from Zenodo.

    Returns the dataset root ``.../data/indoor_object_detection_dataset``.
    """
    import urllib.request
    import zipfile

    data_dir = Path(data_dir or DATA_DIR).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    data_root = data_dir / DATASET_SUBDIR

    zip_path = data_dir / "indoor_object_detection_dataset.zip"

    if data_root.is_dir() and not force:
        if remove_zip:
            remove_dataset_zips(data_dir)
        return data_root

    zenodo_url = (
        "https://zenodo.org/records/2654485/files/"
        "Indoor%20Object%20Detection%20Dataset.zip?download=1"
    )

    try:
        if force or not data_root.is_dir():
            if force or not zip_path.is_file():
                print("Downloading dataset...")
                urllib.request.urlretrieve(zenodo_url, zip_path)
            print("Extracting...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_dir)
            if not data_root.is_dir():
                for candidate in data_dir.iterdir():
                    if candidate.is_dir() and any(candidate.glob("sequence_*")):
                        if candidate != data_root:
                            candidate.rename(data_root)
                        break

        if not data_root.is_dir():
            raise FileNotFoundError(f"Dataset not found at {data_root} after extract")
    finally:
        if remove_zip:
            remove_dataset_zips(data_dir)

    return data_root


def run_dataset_analysis(data_root: Path | str) -> DatasetStats:
    """Print dataset statistics and return the computed :class:`DatasetStats`."""
    data_root = Path(data_root).resolve()
    images = collect_sequence_images(data_root)
    annotations = load_all_annotations(data_root)
    stats = compute_dataset_stats(images, annotations)
    print_dataset_stats(stats, data_root)
    return stats


def prepare_browse_images(
    data_root: Path | str,
    *,
    filter_class: str | None = None,
    filter_split: str | None = None,
    seed: int | None = None,
) -> tuple[list[Path], dict[str, list[Box]]]:
    """Collect images (and annotations) for the interactive viewers."""
    data_root = Path(data_root).resolve()
    images = collect_sequence_images(data_root)
    annotations = load_all_annotations(data_root)
    view_images = images
    if filter_split:
        split_names = load_split_image_names(data_root, filter_split)
        view_images = filter_images_by_split(view_images, split_names)
    if filter_class:
        view_images = filter_images_by_class(view_images, annotations, filter_class)
    order = view_images.copy()
    if seed is not None:
        random.Random(seed).shuffle(order)
    return order, annotations


def browse_dataset(
    data_root: Path | str,
    *,
    seed: int | None = None,
    show_overlays: bool = True,
    filter_class: str | None = None,
    filter_split: str | None = None,
) -> None:
    """Interactive OpenCV viewer (requires a display)."""
    data_root = Path(data_root).resolve()
    view_images, annotations = prepare_browse_images(
        data_root,
        filter_class=filter_class,
        filter_split=filter_split,
        seed=seed,
    )
    browse(
        view_images,
        data_root,
        annotations,
        seed=None,
        show_overlays=show_overlays,
        filter_class=filter_class,
        filter_split=filter_split,
    )


def browse_dataset_notebook(
    data_root: Path | str,
    *,
    seed: int | None = 42,
    show_overlays: bool = True,
) -> None:
    """Interactive dataset browser for Jupyter / Colab (widgets + matplotlib)."""
    try:
        import ipywidgets as widgets
        import matplotlib.pyplot as plt
        import cv2
        from IPython.display import display
    except ImportError as exc:
        raise ImportError(
            "browse_dataset_notebook requires ipywidgets and matplotlib "
            "(install via requirements.txt)."
        ) from exc

    data_root = Path(data_root).resolve()
    state: dict = {
        "idx": 0,
        "images": [],
        "annotations": {},
        "filter_class": None,
        "filter_split": None,
    }

    class_options: list[tuple[str, str | None]] = [("(all classes)", None)]
    class_options.extend((name, name) for name in CLASS_NAMES)
    split_options: list[tuple[str, str | None]] = [("(all images)", None)]
    split_options.extend((name, name) for name in SPLIT_NAMES)

    class_dd = widgets.Dropdown(
        options=class_options,
        description="Class:",
        layout=widgets.Layout(width="220px"),
    )
    split_dd = widgets.Dropdown(
        options=split_options,
        description="Split:",
        layout=widgets.Layout(width="200px"),
    )
    prev_btn = widgets.Button(description="← Previous")
    next_btn = widgets.Button(description="Next →")
    count_label = widgets.Label(value="")
    out = widgets.Output()

    def _status_text(path: Path, idx: int, n: int, boxes: list[Box]) -> str:
        try:
            rel = path.relative_to(data_root)
        except ValueError:
            rel = path.name
        parts = [f"[{idx + 1}/{n}]", str(rel)]
        if state["filter_split"]:
            parts.append(f"split={state['filter_split']}")
        if state["filter_class"]:
            parts.append(f"class={state['filter_class']}")
        if show_overlays:
            nbox = len(boxes)
            parts.append(f"{nbox} box{'es' if nbox != 1 else ''}")
        return "  |  ".join(parts)

    def _show() -> None:
        with out:
            out.clear_output(wait=True)
            images: list[Path] = state["images"]
            if not images:
                print("No images match the current filters.")
                count_label.value = "0 images"
                return

            idx = state["idx"] % len(images)
            state["idx"] = idx
            path = images[idx]
            count_label.value = f"{idx + 1} / {len(images)}"

            bgr = cv2.imread(str(path))
            if bgr is None:
                print(f"Could not read {path}")
                return

            boxes = state["annotations"].get(path.name, []) if show_overlays else []
            fc = state["filter_class"]
            if show_overlays and fc:
                boxes = [b for b in boxes if b.label == fc]
            vis = draw_annotations(bgr, boxes) if (show_overlays and boxes) else bgr
            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

            fig, ax = plt.subplots(figsize=(12, 7))
            ax.imshow(rgb)
            ax.axis("off")
            ax.set_title(_status_text(path, idx, len(images), boxes), fontsize=10)
            fig.tight_layout()
            plt.show()
            plt.close(fig)

    def _reload(_change=None) -> None:
        fc = class_dd.value
        fs = split_dd.value
        state["filter_class"] = fc
        state["filter_split"] = fs
        try:
            images, annotations = prepare_browse_images(
                data_root,
                filter_class=fc,
                filter_split=fs,
                seed=seed,
            )
        except FileNotFoundError as exc:
            with out:
                out.clear_output(wait=True)
                print(exc)
            state["images"] = []
            state["annotations"] = {}
            count_label.value = "split list missing"
            return

        state["images"] = images
        state["annotations"] = annotations
        state["idx"] = 0
        _show()

    def _step(delta: int) -> None:
        if state["images"]:
            state["idx"] = (state["idx"] + delta) % len(state["images"])
            _show()

    prev_btn.on_click(lambda _: _step(-1))
    next_btn.on_click(lambda _: _step(1))
    class_dd.observe(_reload, names="value")
    split_dd.observe(_reload, names="value")

    controls = widgets.HBox([prev_btn, next_btn, class_dd, split_dd, count_label])
    display(controls, out)
    _reload()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Indoor detection dataset: print stats (default) or visualize with --visualize.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Dataset root (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open interactive viewer (random order, bbox overlays)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random shuffle seed for --visualize",
    )
    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="With --visualize: show images without bounding boxes",
    )
    class_filter = parser.add_mutually_exclusive_group()
    class_filter.add_argument(
        "--class",
        dest="filter_class",
        metavar="NAME",
        choices=CLASS_NAMES,
        help=f"With --visualize: only images containing this class ({', '.join(CLASS_NAMES)})",
    )
    for class_name in CLASS_NAMES:
        class_filter.add_argument(
            f"--{class_name}",
            action="store_const",
            const=class_name,
            dest="filter_class",
            help=f"With --visualize: only images containing '{class_name}'",
        )
    split_filter = parser.add_mutually_exclusive_group()
    for split_name in SPLIT_NAMES:
        split_filter.add_argument(
            f"--{split_name}",
            action="store_const",
            const=split_name,
            dest="filter_split",
            help=f"With --visualize: only images listed in {SPLIT_LIST_FILES[split_name]}",
        )
    args = parser.parse_args()
    data_root = args.data_root.resolve()

    if args.filter_class and not args.visualize:
        parser.error("Class filter flags require --visualize.")
    if args.filter_split and not args.visualize:
        parser.error("Split flags (--train, --val, --test) require --visualize.")

    if args.visualize:
        if args.filter_split:
            try:
                load_split_image_names(data_root, args.filter_split)
            except FileNotFoundError as exc:
                parser.error(str(exc))
        browse_dataset(
            data_root,
            seed=args.seed,
            show_overlays=not args.no_overlays,
            filter_class=args.filter_class,
            filter_split=args.filter_split,
        )
    else:
        run_dataset_analysis(data_root)


if __name__ == "__main__":
    main()
