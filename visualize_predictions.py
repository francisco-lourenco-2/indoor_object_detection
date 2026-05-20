#!/usr/bin/env python3
"""
Visualize best / worst YOLO predictions on a dataset split.

Examples are ranked by **per-image contribution to mAP** (COCO IoU grid + precision),
not by prediction confidence.

**Notebook (recommended)** — images render inline under the cell::

    %matplotlib inline
    from visualize_predictions import run_visualization

    run_visualization(
        "indoor_object_detection/original/yolov8n/experiment_2",
        split="test",
        device="0",
    )

**Terminal** — saves montage PNGs (no GUI by default)::

    python visualize_predictions.py \\
        --work-dir indoor_object_detection/original/yolov8n/experiment_2 --test

``!python visualize_predictions.py`` in a notebook only prints text; it cannot
display figures because it runs in a subprocess. Import ``run_visualization`` instead.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# Before OpenCV: avoid Qt/xcb if something tries to open a GUI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent
WORK_DIRS_ROOT = REPO_ROOT / "work_dirs"
DEFAULT_DATA_YAML = REPO_ROOT / "prepared_data" / "yolo_indoor_object_detection" / "indoor.yaml"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# COCO detection: 10 IoU thresholds from 0.50 to 0.95 step 0.05
COCO_IOU_THRESHOLDS = tuple(np.linspace(0.5, 0.95, 10).tolist())
DisplayMode = Literal["auto", "notebook", "save", "gui", "none"]


@dataclass(frozen=True)
class ScoredImage:
    path: Path
    map_score: float  # higher ≈ better contribution to split mAP on this frame
    mean_matched_iou: float
    tp50: int
    fp: int
    fn: int


@dataclass
class VizResult:
    """Outputs from :func:`run_visualization`."""

    work_dir: Path
    split: str
    metrics: dict[str, float]
    per_class: dict[str, dict[str, float]]
    mean_inference_ms: float
    inference_std_ms: float
    n_images: int
    saved_paths: list[Path] = field(default_factory=list)
    figures: list[Any] = field(default_factory=list)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize best/worst YOLO predictions on a dataset split.",
    )
    parser.add_argument("--work-dir", type=str, required=True)
    split = parser.add_mutually_exclusive_group(required=True)
    split.add_argument("--train", action="store_true")
    split.add_argument("--val", action="store_true")
    split.add_argument("--valid", action="store_true")
    split.add_argument("--test", action="store_true")
    parser.add_argument("--data-yaml", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--thumb-size", type=int, default=320)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--show",
        action="store_true",
        help="GUI window (terminal only; use import run_visualization in notebooks)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show YOLO logs and extra progress (default: quiet summary only)",
    )
    return parser.parse_args()


def _split_from_args(args: argparse.Namespace) -> str:
    if args.train:
        return "train"
    if args.test:
        return "test"
    return "val"


def resolve_work_dir(path_str: str | Path) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        candidate = WORK_DIRS_ROOT / p
        if candidate.exists() or (p.parts and (WORK_DIRS_ROOT / p.parts[0]).exists()):
            p = candidate
        else:
            p = REPO_ROOT / p
    return p.resolve()


def resolve_data_yaml(work_dir: Path, cli_yaml: Path | None) -> Path:
    if cli_yaml is not None:
        return cli_yaml.expanduser().resolve()
    meta_path = work_dir / "meta.json"
    if meta_path.is_file():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("data_yaml"):
            return Path(meta["data_yaml"]).resolve()
    return DEFAULT_DATA_YAML.resolve()


def resolve_checkpoint(work_dir: Path) -> Path:
    for ckpt in (
        work_dir / "ckpts" / "best.pt",
        work_dir / "ckpts" / "last.pt",
        work_dir / "yolo_train" / "weights" / "best.pt",
        work_dir / "yolo_train" / "weights" / "last.pt",
    ):
        if ckpt.is_file():
            return ckpt
    raise FileNotFoundError(f"No checkpoint under {work_dir}")


def load_dataset_paths(data_yaml: Path, split: str) -> tuple[Path, Path]:
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    images_dir = (root / cfg[split]).resolve()
    labels_dir = images_dir.parent.parent / "labels" / images_dir.name
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    return images_dir, labels_dir


def list_images(images_dir: Path) -> list[Path]:
    return sorted(
        (p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.name,
    )


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> list[tuple[int, np.ndarray]]:
    if not label_path.is_file():
        return []
    boxes: list[tuple[int, np.ndarray]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:5])
        boxes.append(
            (
                cls,
                np.array(
                    [
                        (cx - bw / 2) * img_w,
                        (cy - bh / 2) * img_h,
                        (cx + bw / 2) * img_w,
                        (cy + bh / 2) * img_h,
                    ],
                    dtype=np.float32,
                ),
            )
        )
    return boxes


def _pred_boxes(result) -> list[tuple[int, np.ndarray, float]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    return [(int(c), xyxy[i].astype(np.float32), float(conf[i])) for i, c in enumerate(cls)]


def _greedy_class_matches(
    gt: list[tuple[int, np.ndarray]],
    pred: list[tuple[int, np.ndarray, float]],
) -> list[tuple[int, int, float]]:
    """
    One-to-one greedy matching by IoU (same class). Prediction confidence is ignored.
    Returns (gt_index, pred_index, iou) for each match.
    """
    candidates: list[tuple[float, int, int]] = []
    for gi, (gc, gb) in enumerate(gt):
        for pi, (pc, pb, _) in enumerate(pred):
            if gc != pc:
                continue
            iou = _box_iou(pb, gb)
            if iou > 0:
                candidates.append((iou, gi, pi))
    candidates.sort(reverse=True)

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((gi, pi, iou))
    return matches


def per_image_map_contribution(
    gt: list[tuple[int, np.ndarray]],
    pred: list[tuple[int, np.ndarray, float]],
) -> tuple[float, float, int, int, int]:
    """
    Per-image proxy for contribution to COCO-style mAP (ranking only; not confidence).

    - **Recall leg**: mean over IoU thresholds 0.5…0.95 of (fraction of GT boxes matched at
      that threshold) — same IoU grid as mAP@0.5:0.95.
    - **Precision leg**: precision at IoU 0.5 (TP predictions / all predictions).
    - **Score**: F1-style combination of those two (high when both localization and
      completeness on this frame are strong).

    Empty image with no GT and no preds scores 1.0; pure FP or pure FN scores 0.0.
    """
    n_gt, n_pred = len(gt), len(pred)
    if n_gt == 0 and n_pred == 0:
        return 1.0, 1.0, 0, 0, 0
    if n_gt == 0:
        return 0.0, 0.0, 0, n_pred, 0
    if n_pred == 0:
        return 0.0, 0.0, 0, 0, n_gt

    matches = _greedy_class_matches(gt, pred)
    recalls = [
        sum(1 for _, _, iou in matches if iou >= thr) / n_gt for thr in COCO_IOU_THRESHOLDS
    ]
    mean_recall = float(np.mean(recalls))
    mean_matched_iou = float(np.mean([m[2] for m in matches])) if matches else 0.0

    tp50 = sum(1 for _, _, iou in matches if iou >= 0.5)
    pred_hit_50 = {pi for _, pi, iou in matches if iou >= 0.5}
    fp = n_pred - len(pred_hit_50)
    fn = n_gt - sum(1 for gi, _, iou in matches if iou >= 0.5)

    precision = tp50 / n_pred
    if mean_recall + precision == 0:
        score = 0.0
    else:
        score = 2 * mean_recall * precision / (mean_recall + precision)

    return score, mean_matched_iou, tp50, fp, fn


def _ultralytics_box_metrics(metrics_obj) -> dict[str, float]:
    out: dict[str, float] = {}
    box = getattr(metrics_obj, "box", None) if metrics_obj is not None else None
    if box is not None:
        for name in ("map", "map50", "map75", "mp", "mr"):
            val = getattr(box, name, None)
            if val is not None:
                try:
                    out[name] = float(val)
                except (TypeError, ValueError):
                    pass
    return out


def _class_names_from_val(data_yaml: Path, metrics_obj) -> dict[int, str]:
    names = getattr(metrics_obj, "names", None) if metrics_obj is not None else None
    if isinstance(names, dict) and names:
        return {int(k): str(v) for k, v in names.items()}
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    raw = cfg.get("names") or {}
    return {int(k): str(v) for k, v in raw.items()}


def _fmt_metric(x, digits: int = 4) -> str:
    try:
        v = float(x)
        if np.isfinite(v):
            return f"{v:.{digits}f}"
    except (TypeError, ValueError):
        pass
    return "?"


def _per_class_metrics(metrics_obj) -> dict[str, dict[str, float]]:
    """Per-class detection metrics from Ultralytics ``model.val()`` (mAP @ 0.5:0.95, AP50, P, R)."""
    box = getattr(metrics_obj, "box", None) if metrics_obj is not None else None
    if box is None:
        return {}

    def _arr(name: str) -> np.ndarray | None:
        val = getattr(box, name, None)
        if val is None:
            return None
        try:
            return np.asarray(val, dtype=np.float64).ravel()
        except (TypeError, ValueError):
            return None

    maps = _arr("maps")
    ap50 = _arr("ap50")
    prec = _arr("p")
    rec = _arr("r")
    n = max(
        len(a) if a is not None else 0
        for a in (maps, ap50, prec, rec)
    )
    if n == 0:
        return {}

    out: dict[str, dict[str, float]] = {}
    for i in range(n):
        row: dict[str, float] = {}
        if maps is not None and i < len(maps):
            row["mAP"] = float(maps[i])
        if ap50 is not None and i < len(ap50):
            row["AP50"] = float(ap50[i])
        if prec is not None and i < len(prec):
            row["precision"] = float(prec[i])
        if rec is not None and i < len(rec):
            row["recall"] = float(rec[i])
        out[str(i)] = row
    return out


@contextlib.contextmanager
def _suppress_yolo_output():
    """Hide Ultralytics progress bars and console logs during val/predict."""
    loggers = [
        logging.getLogger(name)
        for name in ("ultralytics", "ultralytics.nn", "ultralytics.utils")
    ]
    saved_levels = [(lg, lg.level) for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.ERROR)

    old_yolo_verbose = os.environ.get("YOLO_VERBOSE")
    os.environ["YOLO_VERBOSE"] = "False"

    ultra_logger = None
    old_ultra_level = None
    try:
        from ultralytics.utils import LOGGER as ultra_logger

        old_ultra_level = ultra_logger.level
        ultra_logger.setLevel(logging.ERROR)
    except Exception:
        ultra_logger = None

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        try:
            yield
        finally:
            for lg, level in saved_levels:
                lg.setLevel(level)
            if old_yolo_verbose is None:
                os.environ.pop("YOLO_VERBOSE", None)
            else:
                os.environ["YOLO_VERBOSE"] = old_yolo_verbose
            if ultra_logger is not None and old_ultra_level is not None:
                ultra_logger.setLevel(old_ultra_level)


def _print_results_summary(
    class_names: dict[int, str],
    per_class: dict[str, dict[str, float]],
    *,
    split: str,
    overall: dict[str, float],
    mean_inference_ms: float,
) -> None:
    header = f"{'class':<20} {'mAP':>8} {'AP50':>8}"
    print(f"split: {split}")
    print(header)
    print("-" * len(header))
    print(
        f"{'all (overall)':<20} "
        f"{_fmt_metric(overall.get('mAP')):>8} "
        f"{_fmt_metric(overall.get('AP50')):>8}"
    )
    print("-" * len(header))
    for idx in sorted(class_names):
        name = class_names[idx]
        row = per_class.get(str(idx), per_class.get(idx, {}))
        print(
            f"{name:<20} "
            f"{_fmt_metric(row.get('mAP')):>8} "
            f"{_fmt_metric(row.get('AP50')):>8}"
        )
    if math.isfinite(mean_inference_ms):
        print(f"\nmean inference time: {mean_inference_ms:.2f} ms/image")
    else:
        print("\nmean inference time: n/a")
    print()


def _read_meta(work_dir: Path) -> dict:
    meta_path = work_dir / "meta.json"
    if not meta_path.is_file():
        return {}
    import json

    return json.loads(meta_path.read_text(encoding="utf-8"))


def _plot_prediction(model, img_path: Path, conf: float, imgsz: int, device: str) -> np.ndarray:
    results = model.predict(
        str(img_path), conf=conf, imgsz=imgsz, device=device, verbose=False
    )
    if not results:
        raise RuntimeError(f"No prediction result for {img_path}")
    return results[0].plot()


def _resize_rgb(bgr: np.ndarray, max_side: int) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = min(max_side / w, max_side / h, 1.0)
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return rgb


def _figure_montage(
    panels: list[tuple[np.ndarray, str]],
    *,
    title: str,
    cols: int = 4,
):
    import matplotlib.pyplot as plt

    n = len(panels)
    cols = min(max(cols, 1), max(n, 1))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.2))
    axes_arr = np.atleast_1d(axes).ravel()

    for ax in axes_arr:
        ax.axis("off")

    for ax, (rgb, cap) in zip(axes_arr, panels):
        ax.imshow(rgb)
        ax.set_title(cap, fontsize=8)
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def _in_ipython() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _resolve_display_mode(requested: DisplayMode) -> DisplayMode:
    if requested != "auto":
        return requested
    return "notebook" if _in_ipython() else "save"


def _resolve_save_dir(
    work_dir: Path,
    split: str,
    cli_dir: Path | None,
    no_save: bool,
    display_mode: DisplayMode,
) -> Path | None:
    if no_save:
        return None
    if cli_dir is not None:
        return cli_dir.expanduser().resolve()
    if display_mode == "save":
        return (work_dir / f"viz_{split}").resolve()
    if display_mode == "notebook":
        return None
    if display_mode == "auto":
        return None if _in_ipython() else (work_dir / f"viz_{split}").resolve()
    return (work_dir / f"viz_{split}").resolve() if display_mode == "gui" else None


def _emit_figure(
    fig,
    *,
    heading: str,
    save_path: Path | None,
    display_mode: DisplayMode,
    store_figures: bool,
    out_figures: list,
) -> None:
    import matplotlib.pyplot as plt

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path.resolve()}")

    if display_mode == "notebook":
        from IPython.display import HTML, display

        if heading:
            display(HTML(f"<h4>{heading}</h4>"))
        display(fig)
        if not store_figures:
            plt.close(fig)
        else:
            out_figures.append(fig)
        return

    if display_mode == "gui" and _has_display():
        if heading:
            print(heading)
        plt.show()
        return

    if store_figures:
        out_figures.append(fig)
    else:
        plt.close(fig)

    if display_mode not in ("save", "none") and save_path is None and heading:
        print(f"{heading} — no display; use display='notebook' or pass save_dir.")


def run_visualization(
    work_dir: str | Path,
    *,
    split: str = "test",
    data_yaml: Path | None = None,
    conf: float = 0.25,
    imgsz: int | None = None,
    device: str | None = None,
    top_k: int = 8,
    thumb_size: int = 320,
    save_dir: Path | None = None,
    no_save: bool = False,
    display: DisplayMode = "auto",
    store_figures: bool = False,
    quiet: bool = True,
) -> VizResult:
    """
    Run val mAP, per-image scoring, and show/save best & worst prediction montages.

    In Jupyter/Colab/Cursor notebooks, call this directly (with ``%matplotlib inline``)
    so figures appear under the cell. Set ``display='notebook'`` to force inline output.
    """
    split = {"valid": "val"}.get(split, split)
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be train, val, or test (got {split!r})")

    work_dir = resolve_work_dir(work_dir)
    if not work_dir.is_dir():
        raise FileNotFoundError(f"Work directory not found: {work_dir}")

    meta = _read_meta(work_dir)
    data_yaml = resolve_data_yaml(work_dir, data_yaml)
    ckpt = resolve_checkpoint(work_dir)
    images_dir, labels_dir = load_dataset_paths(data_yaml, split)
    images = list_images(images_dir)
    if not images:
        raise RuntimeError(f"No images in {images_dir}")

    imgsz = imgsz or int(meta.get("imgsz", 640))
    device = device if device is not None else str(meta.get("device", "0"))
    display_mode = _resolve_display_mode(display)
    out_save_dir = _resolve_save_dir(work_dir, split, save_dir, no_save, display_mode)

    from ultralytics import YOLO

    if not quiet:
        print(f"Work dir:   {work_dir}")
        print(f"Checkpoint: {ckpt}")
        print(f"Split:      {split} ({len(images)} images)")
        print(f"Display:    {display_mode}")
        print()

    with _suppress_yolo_output():
        model = YOLO(str(ckpt))
        val_metrics = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        )

    box = _ultralytics_box_metrics(val_metrics)
    class_names = _class_names_from_val(data_yaml, val_metrics)
    per_class_raw = _per_class_metrics(val_metrics)
    per_class = {
        class_names.get(int(idx), idx): vals for idx, vals in per_class_raw.items()
    }

    metrics = {
        "mAP": box.get("map", float("nan")),
        "AP50": box.get("map50", float("nan")),
        "AP75": box.get("map75", float("nan")),
        "precision": box.get("mp", float("nan")),
        "recall": box.get("mr", float("nan")),
    }

    scored: list[ScoredImage] = []
    timings_ms: list[float] = []

    with _suppress_yolo_output():
        for i, img_path in enumerate(images):
            t0 = time.perf_counter()
            results = model.predict(
                str(img_path), conf=conf, imgsz=imgsz, device=device, verbose=False
            )
            timings_ms.append((time.perf_counter() - t0) * 1000.0)

            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            gt = _load_gt_boxes(labels_dir / f"{img_path.stem}.txt", w, h)
            pred = _pred_boxes(results[0]) if results else []
            map_score, mean_iou, tp50, fp, fn = per_image_map_contribution(gt, pred)
            scored.append(
                ScoredImage(
                    path=img_path,
                    map_score=map_score,
                    mean_matched_iou=mean_iou,
                    tp50=tp50,
                    fp=fp,
                    fn=fn,
                )
            )

            if not quiet and ((i + 1) % 50 == 0 or i + 1 == len(images)):
                print(f"  {i + 1}/{len(images)}")

    arr = np.array(timings_ms, dtype=np.float64)
    mean_ms = float(arr.mean()) if len(arr) else float("nan")
    std_ms = float(arr.std()) if len(arr) else float("nan")

    if quiet:
        _print_results_summary(
            class_names,
            per_class_raw,
            split=split,
            overall=metrics,
            mean_inference_ms=mean_ms,
        )
    else:
        print(
            f"mAP={metrics['mAP']:.4f} | AP50={metrics['AP50']:.4f} | "
            f"AP75={metrics['AP75']:.4f} | precision={metrics['precision']:.4f} | "
            f"recall={metrics['recall']:.4f}"
        )
        print()
        _print_results_summary(
            class_names,
            per_class_raw,
            split=split,
            overall=metrics,
            mean_inference_ms=mean_ms,
        )

    scored.sort(key=lambda s: s.map_score)
    k = min(top_k, len(scored))
    worst = scored[:k]
    best = list(reversed(scored[-k:]))

    def build_panels(items: list[ScoredImage]) -> list[tuple[np.ndarray, str]]:
        panels: list[tuple[np.ndarray, str]] = []
        for s in items:
            with _suppress_yolo_output():
                bgr = _plot_prediction(model, s.path, conf, imgsz, device)
            rgb = _resize_rgb(bgr, thumb_size)
            cap = (
                f"{s.path.name}\n"
                f"mAP score={s.map_score:.3f}  meanIoU={s.mean_matched_iou:.2f}\n"
                f"TP@0.5={s.tp50} FP={s.fp} FN={s.fn}"
            )
            panels.append((rgb, cap))
        return panels

    if not quiet:
        print(
            f"Building {k} worst and {k} best montages "
            f"(ranked by per-image mAP contribution, not confidence)..."
        )
    saved: list[Path] = []
    figures: list[Any] = []

    worst_fig = _figure_montage(
        build_panels(worst),
        title=f"Worst {k} — {split} (lowest per-image mAP contribution)",
    )
    best_fig = _figure_montage(
        build_panels(best),
        title=f"Best {k} — {split} (highest per-image mAP contribution)",
    )

    worst_png = (out_save_dir / "montage_worst.png") if out_save_dir else None
    best_png = (out_save_dir / "montage_best.png") if out_save_dir else None

    _emit_figure(
        worst_fig,
        heading=f"Worst {k} predictions",
        save_path=worst_png,
        display_mode=display_mode,
        store_figures=store_figures,
        out_figures=figures,
    )
    _emit_figure(
        best_fig,
        heading=f"Best {k} predictions",
        save_path=best_png,
        display_mode=display_mode,
        store_figures=store_figures,
        out_figures=figures,
    )

    if worst_png:
        saved.append(worst_png)
    if best_png:
        saved.append(best_png)

    return VizResult(
        work_dir=work_dir,
        split=split,
        metrics=metrics,
        per_class=per_class,
        mean_inference_ms=mean_ms,
        inference_std_ms=std_ms,
        n_images=len(images),
        saved_paths=saved,
        figures=figures,
    )


def main() -> None:
    args = _parse_args()
    display: DisplayMode = "gui" if args.show else "auto"
    run_visualization(
        args.work_dir,
        split=_split_from_args(args),
        data_yaml=args.data_yaml,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        top_k=args.top_k,
        thumb_size=args.thumb_size,
        save_dir=args.save_dir,
        no_save=args.no_save,
        display=display,
        quiet=not args.verbose,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
