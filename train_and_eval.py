#!/usr/bin/env python3
"""
DocuSketch indoor object detection — YOLOv8 train/val/test with experiment logging.

Uses the prepared Ultralytics dataset (``prepared_data/yolo_indoor_object_detection/indoor.yaml``)
and logs runs under ``work_dirs/<dataset>/<lv2>/<lv3>/experiment_N/`` via ``exp_logging/``.
"""

from __future__ import annotations

import argparse
import builtins
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
DATASET_NAME = "indoor_object_detection"
DEFAULT_MODEL = "yolov8n.pt"
ULTRALYTICS_RUN_NAME = "yolo_train"

# =======================
# ====== GLOBALS =========
# =======================
# Dataset / training
DATA_YAML = REPO_ROOT / "prepared_data" / "yolo_indoor_object_detection" / "indoor.yaml"
EPOCHS = 20
IMGSZ = 640
BATCH_SIZE = 16  # -1 = Ultralytics AutoBatch
DEVICE = 0  # GPU index, or "cpu"
PATIENCE = 20  # early-stopping patience (epochs)

# Augmentations (0 disables) (ultralytics defaults)
HSV_H = 0.015
HSV_S = 0.7
HSV_V = 0.4
DEGREES = 0.0
TRANSLATE = 0.1
SCALE = 0.5
SHEAR = 0.0
PERSPECTIVE = 0.0
FLIPUD = 0.0
FLIPLR = 0.5
MOSAIC = 1.0
MIXUP = 0.0
CUTMIX = 0.0
CLOSE_MOSAIC = 10

# Reproducibility / logging
SEED = 42
CONF_VIZ = 0.25  # confidence threshold for overlay images
N_OVERLAY = 230  # random prediction panels per split
# False = let Ultralytics print its native epoch table (metrics still logged to exp_logging).
LOG_CUSTOM_EPOCH_SUMMARY = True

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def setup_logging(work_dir: Path) -> None:
    from exp_logging.io_utils import ensure_dir

    ensure_dir(work_dir)
    log_file = work_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logging.info("Logging to: %s", log_file)


def build_training_answers(
    *,
    data_variation: str = "default",
    goal: str | None = None,
    changed: str | None = None,
    constraints: str | None = None,
    hyperparameters: str | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Post-training report Q&A (level 4). Override any field; ``extra`` merges additional keys."""
    answers = {
        "What was the main goal of this experiment?": goal
        or "Train YOLOv8 on the indoor detection dataset.",
        "What changed compared to the baseline/previous best?": changed
        or f"Data variation '{data_variation}' (YOLO export with ILP stratified splits).",
        "Any constraints, issues or edge cases worth noting?": constraints or "None noted.",
        "Which hyperparameters changed and why?": hyperparameters
        or "See meta.json for hyperparameters.",
    }
    if extra:
        answers.update(extra)
    return answers


def default_training_answers(data_variation: str = "default") -> dict[str, str]:
    """Default post-training Q&A for notebook / non-interactive runs (level 4)."""
    return build_training_answers(data_variation=data_variation)


def _patch_non_interactive(lv3_name: str) -> None:
    replies = iter(
        [
            "4",
            "default",
            lv3_name,
            "y",
            "Train YOLOv8 on the indoor detection dataset.",
            "Prepared YOLO export with ILP stratified splits.",
            "None noted.",
            "See meta.json for hyperparameters.",
        ]
    )

    def _fake_input(prompt: str = "") -> str:  # noqa: ARG001
        try:
            return next(replies)
        except StopIteration:
            return ""

    builtins.input = _fake_input  # type: ignore[assignment]


def _metric_value(metrics: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in metrics and metrics[key] is not None:
            try:
                v = float(metrics[key])
                if np.isfinite(v):
                    return v
            except (TypeError, ValueError):
                pass
    return float("nan")


def _fmt_metric(value: float, digits: int = 4) -> str:
    if np.isfinite(value):
        return f"{value:.{digits}f}"
    return "n/a"


def _val_loss_from_metrics(metrics: dict[str, Any]) -> float:
    """Total validation loss (box + cls + dfl) logged by Ultralytics during training."""
    parts = [
        _metric_value(metrics, "val/box_loss"),
        _metric_value(metrics, "val/cls_loss"),
        _metric_value(metrics, "val/dfl_loss"),
    ]
    finite = [v for v in parts if np.isfinite(v)]
    if finite:
        return float(sum(finite))
    return _metric_value(metrics, "val/loss", "loss")


def _val_loss_from_validator(trainer) -> float:
    """Fallback when ``trainer.metrics`` omits val loss keys (e.g. final eval callback)."""
    validator = getattr(trainer, "validator", None)
    if validator is None:
        return float("nan")
    loss = getattr(validator, "loss", None)
    if loss is None:
        return float("nan")
    try:
        total = float(loss.sum()) if hasattr(loss, "sum") else float(loss)
        n_batches = len(getattr(validator, "dataloader", []) or [])
        if n_batches <= 0:
            return float("nan")
        return total / n_batches
    except (TypeError, ValueError):
        return float("nan")


def _validator_box_map75(trainer) -> float:
    """mAP@0.75 is not in ``trainer.metrics`` during training; read from the validator."""
    validator = getattr(trainer, "validator", None)
    if validator is None:
        return float("nan")
    box = getattr(getattr(validator, "metrics", None), "box", None)
    if box is None:
        return float("nan")
    try:
        v = float(box.map75)
        return v if np.isfinite(v) else float("nan")
    except (TypeError, ValueError, AttributeError):
        return float("nan")


def _collect_epoch_metrics(trainer) -> dict[str, float]:
    """Metrics available at ``on_fit_epoch_end`` (after in-loop validation)."""
    metrics = trainer.metrics or {}
    map_all = _metric_value(
        metrics,
        "metrics/mAP50-95(B)",
        "metrics/mAP50-95",
        "val/mAP50-95(B)",
    )
    map50 = _metric_value(metrics, "metrics/mAP50(B)", "metrics/mAP50", "val/mAP50(B)")
    map75 = _validator_box_map75(trainer)
    if not np.isfinite(map75):
        map75 = _metric_value(metrics, "metrics/mAP75(B)", "metrics/mAP75", "val/mAP75(B)")

    return {
        "val_loss": _val_loss_from_metrics(metrics),
        "map": map_all,
        "map50": map50,
        "map75": map75,
        "precision": _metric_value(metrics, "metrics/precision(B)", "metrics/precision"),
        "recall": _metric_value(metrics, "metrics/recall(B)", "metrics/recall"),
    }


class ExpLoggingCallback:
    """Push Ultralytics trainer metrics into ``ScalarTracker`` + ``train.log``."""

    def __init__(self, tracker, logger: logging.Logger, *, log_epoch_summary: bool = LOG_CUSTOM_EPOCH_SUMMARY):
        self.tracker = tracker
        self.logger = logger
        self.log_epoch_summary = log_epoch_summary
        self._last_logged_epoch = 0

    def on_train_epoch_end(self, trainer) -> None:
        epoch = int(trainer.epoch) + 1
        tloss = trainer.tloss
        if tloss is None:
            return
        try:
            loss = float(tloss.sum()) if hasattr(tloss, "sum") else float(tloss)
        except Exception:
            return
        if np.isfinite(loss):
            self.tracker.log_scalar("train/loss", loss, epoch)

    def on_fit_epoch_end(self, trainer) -> None:
        epoch = int(trainer.epoch) + 1
        if epoch <= self._last_logged_epoch:
            return

        m = _collect_epoch_metrics(trainer)
        if not np.isfinite(m["val_loss"]):
            m["val_loss"] = _val_loss_from_validator(trainer)

        if np.isfinite(m["val_loss"]):
            self.tracker.log_scalar("val/loss", m["val_loss"], epoch)
        if np.isfinite(m["map"]):
            self.tracker.log_scalar("val/mAP", m["map"], epoch)
            self.tracker.log_scalar("val/miou", m["map"], epoch)
        if np.isfinite(m["map50"]):
            self.tracker.log_scalar("val/AP50", m["map50"], epoch)
        if np.isfinite(m["map75"]):
            self.tracker.log_scalar("val/AP75", m["map75"], epoch)
        if np.isfinite(m["precision"]):
            self.tracker.log_scalar("val/precision", m["precision"], epoch)
        if np.isfinite(m["recall"]):
            self.tracker.log_scalar("val/recall", m["recall"], epoch)

        lr = float("nan")
        try:
            if trainer.optimizer is not None:
                lr = float(trainer.optimizer.param_groups[0]["lr"])
        except Exception:
            pass
        if np.isfinite(lr):
            self.tracker.log_scalar("opt/lr", lr, epoch)

        train_loss = "n/a"
        if trainer.tloss is not None:
            try:
                train_loss = f"{float(trainer.tloss.sum()):.4f}"
            except Exception:
                pass

        if self.log_epoch_summary:
            self.logger.info(
                "Epoch %03d | train_loss=%s | val_loss=%s | val mAP=%s AP50=%s AP75=%s",
                epoch,
                train_loss,
                _fmt_metric(m["val_loss"]),
                _fmt_metric(m["map"]),
                _fmt_metric(m["map50"]),
                _fmt_metric(m["map75"]),
            )
        self._last_logged_epoch = epoch


def _ultralytics_box_metrics(metrics_obj: Any) -> dict[str, float]:
    """Extract detection metrics from Ultralytics ``model.val()`` return value."""
    out: dict[str, float] = {}
    if metrics_obj is None:
        return out
    box = getattr(metrics_obj, "box", None)
    if box is not None:
        for name in ("map", "map50", "map75", "mp", "mr"):
            val = getattr(box, name, None)
            if val is not None:
                try:
                    out[name] = float(val)
                except (TypeError, ValueError):
                    pass
        return out
    if isinstance(metrics_obj, dict):
        for k, v in metrics_obj.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def _evaluate_split(model, data_yaml: Path, split: str, logger: logging.Logger) -> dict[str, float]:
    logger.info("Evaluating on split=%s ...", split)
    metrics = model.val(data=str(data_yaml), split=split, verbose=False)
    box = _ultralytics_box_metrics(metrics)
    logger.info(
        "[%s] mAP=%.4f | AP50=%.4f | AP75=%.4f | precision=%.4f | recall=%.4f",
        split.upper(),
        box.get("map", float("nan")),
        box.get("map50", float("nan")),
        box.get("map75", float("nan")),
        box.get("mp", float("nan")),
        box.get("mr", float("nan")),
    )
    return box


def _log_test_summary(logger: logging.Logger, box: dict[str, float]) -> None:
    mAP = box.get("map", float("nan"))
    ap50 = box.get("map50", float("nan"))
    ap75 = box.get("map75", float("nan"))
    prec = box.get("mp", float("nan"))
    rec = box.get("mr", float("nan"))
    logger.info(
        "[TEST] mAP=%.4f | AP50=%.4f | AP75=%.4f | precision=%.4f | recall=%.4f",
        mAP,
        ap50,
        ap75,
        prec,
        rec,
    )


def _save_results_json(work_dir: Path, val_box: dict[str, float], test_box: dict[str, float]) -> None:
    from exp_logging.io_utils import write_json

    def _pack(box: dict[str, float]) -> dict[str, float]:
        return {
            "mAP": box.get("map", float("nan")),
            "AP50": box.get("map50", float("nan")),
            "AP75": box.get("map75", float("nan")),
            "precision": box.get("mp", float("nan")),
            "recall": box.get("mr", float("nan")),
        }

    write_json({"val": _pack(val_box), "test": _pack(test_box)}, work_dir / "results.json")


def _copy_checkpoint(run_dir: Path, work_dir: Path) -> Path | None:
    src = run_dir / "weights" / "best.pt"
    if not src.is_file():
        src = run_dir / "weights" / "last.pt"
    if not src.is_file():
        return None
    ckpt_dir = work_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dst = ckpt_dir / "best.pt"
    shutil.copy2(src, dst)
    last = run_dir / "weights" / "last.pt"
    if last.is_file():
        shutil.copy2(last, ckpt_dir / "last.pt")
    return dst


def _save_prediction_overlays(
    model,
    images_dir: Path,
    out_dir: Path,
    *,
    n_samples: int = 12,
    conf: float = 0.25,
    seed: int = 42,
) -> None:
    """Save side-by-side style prediction panels for qualitative review."""
    from exp_logging.io_utils import ensure_dir

    ensure_dir(out_dir)
    images = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )
    if not images:
        logging.warning("No images in %s for overlays.", images_dir)
        return

    rng = random.Random(seed)
    if n_samples > 0 and len(images) > n_samples:
        images = rng.sample(images, n_samples)

    for idx, img_path in enumerate(images):
        results = model.predict(str(img_path), conf=conf, verbose=False)
        if not results:
            continue
        plotted = results[0].plot()
        out_path = out_dir / f"sample_{idx:03d}__{img_path.name}"
        cv2.imwrite(str(out_path), plotted)


def build_yolo_train_kwargs(work_dir: Path, data_yaml: Path) -> dict[str, Any]:
    """Ultralytics ``model.train()`` kwargs from GLOBALS (override via CLI in ``main``)."""
    device = DEVICE
    if isinstance(device, str) and device.strip() == "":
        device = None
    return dict(
        data=str(data_yaml.resolve()),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        device=device,
        project=str(work_dir.resolve()),
        name=ULTRALYTICS_RUN_NAME,
        exist_ok=True,
        pretrained=True,
        patience=PATIENCE,
        save=True,
        plots=True,
        verbose=True,
        seed=SEED,
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        shear=SHEAR,
        perspective=PERSPECTIVE,
        flipud=FLIPUD,
        fliplr=FLIPLR,
        mosaic=MOSAIC,
        mixup=MIXUP,
        cutmix=CUTMIX,
        close_mosaic=CLOSE_MOSAIC,
    )


def train_yolov8(
    work_dir: Path,
    tracker,
    *,
    data_yaml: Path,
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    patience: int,
    conf_viz: float,
    n_overlay: int,
    log_epoch_summary: bool = LOG_CUSTOM_EPOCH_SUMMARY,
) -> None:
    from ultralytics import YOLO

    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}\n"
            "Run: python prepare_yolo_dataset.py"
        )

    yaml_path = data_yaml.resolve()
    dataset_root = yaml_path.parent
    logging.info("YOLO data yaml: %s", yaml_path)

    model = YOLO(model_name)
    cb = ExpLoggingCallback(tracker, logging.getLogger(), log_epoch_summary=log_epoch_summary)
    model.add_callback("on_train_epoch_end", cb.on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", cb.on_fit_epoch_end)

    train_kwargs = build_yolo_train_kwargs(work_dir, yaml_path)
    # CLI / function call overrides (when not None / empty)
    train_kwargs.update(
        {
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "patience": patience,
        }
    )
    if device not in ("", None):
        train_kwargs["device"] = device
    logging.info("Starting Ultralytics training: %s", train_kwargs)
    model.train(**train_kwargs)

    run_dir = Path(getattr(model.trainer, "save_dir", work_dir / ULTRALYTICS_RUN_NAME))
    if not run_dir.is_dir():
        run_dir = work_dir / ULTRALYTICS_RUN_NAME
    logging.info("Ultralytics save_dir: %s", run_dir)
    best_ckpt = _copy_checkpoint(run_dir, work_dir)
    if best_ckpt is None:
        logging.warning("No best/last checkpoint found under %s", run_dir)
    else:
        logging.info("Best weights copied to %s", best_ckpt)
        model = YOLO(str(best_ckpt))

    val_box = _evaluate_split(model, yaml_path, "val", logging.getLogger())
    test_box = _evaluate_split(model, yaml_path, "test", logging.getLogger())
    _log_test_summary(logging.getLogger(), test_box)
    _save_results_json(work_dir, val_box, test_box)

    _save_prediction_overlays(
        model,
        dataset_root / "images" / "train",
        work_dir / "overlays_train",
        n_samples=n_overlay,
        conf=conf_viz,
    )
    _save_prediction_overlays(
        model,
        dataset_root / "images" / "val",
        work_dir / "overlays_val",
        n_samples=n_overlay,
        conf=conf_viz,
    )
    _save_prediction_overlays(
        model,
        dataset_root / "images" / "test",
        work_dir / "overlays_test",
        n_samples=n_overlay,
        conf=conf_viz,
    )

    args_path = work_dir / "train_args.json"
    args_path.write_text(
        json.dumps(
            {
                **{k: v for k, v in train_kwargs.items() if k not in ("project", "name")},
                "model": model_name,
                "data_yaml": str(yaml_path),
                "ultralytics_run": str(run_dir),
            },
            indent=2,
        )
    )
    logging.info("Training complete. Ultralytics run: %s", run_dir)


@dataclass
class TrainingResult:
    work_dir: Path
    meta: dict


def run_training(
    data_yaml: Path | str | None = None,
    *,
    model: str = DEFAULT_MODEL,
    epochs: int = EPOCHS,
    imgsz: int = IMGSZ,
    batch: int = BATCH_SIZE,
    device: str | int | None = DEVICE,
    patience: int = PATIENCE,
    conf_viz: float = CONF_VIZ,
    n_overlay: int = N_OVERLAY,
    log_epoch_summary: bool = LOG_CUSTOM_EPOCH_SUMMARY,
    dataset_name: str = DATASET_NAME,
    data_variation: str | None = None,
    lv2_name: str = "default",
    lv3_name: str | None = None,
    answers: dict[str, str] | None = None,
    interactive: bool = False,
    build_reports: bool = False,
) -> TrainingResult:
    """
    Train YOLOv8 with experiment logging.

    Notebook / API: set ``data_variation`` (Lv2 folder under ``work_dirs/<dataset>/``) and
    optional ``answers``. Dataset name, network tag (from ``model``), and ``experiment_N``
    are inferred automatically. Default ``build_reports=False`` skips REPORT.md / rollups
    (no OpenAI calls; suitable for Colab).

    Terminal: ``interactive=True`` prompts for experiment level and Lv2/Lv3 names.
    Pass ``build_reports=True`` locally when ``OPENAI_API_KEY`` is set and you want reports.

    Returns the saved experiment directory under ``work_dirs/``.
    """
    from exp_logging.io_utils import write_json
    from exp_logging.manager import (
        ask_experiment_level_and_context,
        begin_experiment_session,
        finalize_experiment_session,
        register_sigint_handler,
    )
    from exp_logging.tracker import ScalarTracker

    os.environ.setdefault("MPLBACKEND", "Agg")

    yaml_path = Path(data_yaml or DATA_YAML).resolve()
    lv3 = lv3_name or Path(model).stem.replace(".", "_")
    lv2 = (data_variation if data_variation is not None else lv2_name).strip() or "default"
    device_str = str(device) if device is not None and device != "" else str(DEVICE)

    if interactive:
        ctx = ask_experiment_level_and_context(
            dataset_name=dataset_name,
            default_lv3_model_tag=lv3,
        )
        meta = {
            "level": ctx["level"],
            "dataset": ctx["dataset"],
            "lv2_name": ctx["lv2_name"],
            "lv3_name": ctx["lv3_name"],
        }
        lv2 = meta["lv2_name"]
        lv3 = meta["lv3_name"]
    else:
        meta = {
            "level": 4,
            "dataset": dataset_name,
            "lv2_name": lv2,
            "lv3_name": lv3,
        }

    meta.update(
        {
            "build_reports": build_reports,
            "model": model,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "device": device_str,
            "data_yaml": str(yaml_path),
            "framework": "ultralytics-yolov8",
            "seed": SEED,
            "augment": {
                "hsv_h": HSV_H,
                "hsv_s": HSV_S,
                "hsv_v": HSV_V,
                "degrees": DEGREES,
                "translate": TRANSLATE,
                "scale": SCALE,
                "shear": SHEAR,
                "perspective": PERSPECTIVE,
                "flipud": FLIPUD,
                "fliplr": FLIPLR,
                "mosaic": MOSAIC,
                "mixup": MIXUP,
                "cutmix": CUTMIX,
                "close_mosaic": CLOSE_MOSAIC,
            },
        }
    )

    session = begin_experiment_session(meta)
    register_sigint_handler(session)
    work_dir = session.temp_dir
    setup_logging(work_dir)
    write_json(meta, work_dir / "meta.json")
    tracker = ScalarTracker(work_dir)

    device_arg = device_str if device_str else ""
    try:
        train_yolov8(
            work_dir,
            tracker,
            data_yaml=yaml_path,
            model_name=model,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device_arg,
            patience=patience,
            conf_viz=conf_viz,
            n_overlay=n_overlay,
            log_epoch_summary=log_epoch_summary,
        )
    except KeyboardInterrupt:
        logging.info("Interrupted; finalize via SIGINT handler.")
        raise

    saved_dir = finalize_experiment_session(
        session,
        keep=True,
        answers=None if interactive else {},
        build_reports=build_reports,
        exit_on_finish=False,
    )
    assert saved_dir is not None
    return TrainingResult(work_dir=saved_dir, meta=session.meta)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on indoor detection (with exp_logging).")
    parser.add_argument("--data", type=Path, default=DATA_YAML, help="Path to indoor.yaml")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ultralytics model checkpoint or yaml")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--device",
        type=str,
        default=str(DEVICE) if DEVICE != "" and DEVICE is not None else "",
        help="cuda device, e.g. 0 or cpu (default: DEVICE global)",
    )
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Early-stopping patience (epochs)")
    parser.add_argument("--conf-viz", type=float, default=CONF_VIZ, help="Confidence threshold for overlay images")
    parser.add_argument("--n-overlay", type=int, default=N_OVERLAY, help="Random overlay images per split")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip prompts (auto experiment metadata + keep logs)",
    )
    parser.add_argument(
        "--lv3-name",
        type=str,
        default=None,
        help="Network tag for work_dirs (default: derived from --model)",
    )
    parser.add_argument(
        "--log-epoch-summary",
        action="store_true",
        help="Print our one-line epoch summary (default: off; Ultralytics table only)",
    )
    parser.add_argument(
        "--reports",
        action="store_true",
        help="Generate REPORT.md and rollups (requires OPENAI_API_KEY for LLM sections)",
    )
    args = parser.parse_args()

    if args.non_interactive:
        lv3 = args.lv3_name or Path(args.model).stem.replace(".", "_")
        _patch_non_interactive(lv3)
        run_training(
            args.data,
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device if args.device != "" else DEVICE,
            patience=args.patience,
            conf_viz=args.conf_viz,
            n_overlay=args.n_overlay,
            log_epoch_summary=args.log_epoch_summary or LOG_CUSTOM_EPOCH_SUMMARY,
            lv3_name=lv3,
            build_reports=args.reports,
        )
        return

    os.environ.setdefault("MPLBACKEND", "Agg")

    from exp_logging.io_utils import write_json
    from exp_logging.manager import (
        ask_experiment_level_and_context,
        begin_experiment_session,
        finalize_experiment_session,
        register_sigint_handler,
    )
    from exp_logging.tracker import ScalarTracker

    default_lv3 = args.lv3_name or Path(args.model).stem.replace(".", "_")
    meta = ask_experiment_level_and_context(
        dataset_name=DATASET_NAME,
        default_lv3_model_tag=default_lv3,
    )
    meta.update(
        {
            "model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device if args.device else DEVICE,
            "data_yaml": str(args.data.resolve()),
            "framework": "ultralytics-yolov8",
            "seed": SEED,
            "augment": {
                "hsv_h": HSV_H,
                "hsv_s": HSV_S,
                "hsv_v": HSV_V,
                "degrees": DEGREES,
                "translate": TRANSLATE,
                "scale": SCALE,
                "shear": SHEAR,
                "perspective": PERSPECTIVE,
                "flipud": FLIPUD,
                "fliplr": FLIPLR,
                "mosaic": MOSAIC,
                "mixup": MIXUP,
                "cutmix": CUTMIX,
                "close_mosaic": CLOSE_MOSAIC,
            },
        }
    )

    session = begin_experiment_session(meta)
    register_sigint_handler(session)
    work_dir = session.temp_dir
    setup_logging(work_dir)
    write_json(meta, work_dir / "meta.json")
    tracker = ScalarTracker(work_dir)

    device = args.device if args.device != "" else None
    try:
        train_yolov8(
            work_dir,
            tracker,
            data_yaml=args.data.resolve(),
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device if device is not None else "",
            patience=args.patience,
            conf_viz=args.conf_viz,
            n_overlay=args.n_overlay,
            log_epoch_summary=args.log_epoch_summary or LOG_CUSTOM_EPOCH_SUMMARY,
        )
    except KeyboardInterrupt:
        logging.info("Interrupted; finalize prompt will run via SIGINT handler.")
        raise

    finalize_experiment_session(session, build_reports=True)


if __name__ == "__main__":
    main()
