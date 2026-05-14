#!/usr/bin/env python3
"""
DocuSketch take-home: indoor object detection — train, validate (mAP), qualitative examples.

This entrypoint will hold the full training and evaluation loop. For now it wires the same
experiment logging used in ``OO_segmentation`` (``exp_logging/``): runs live under
``work_dirs/<dataset>/<lv2>/<lv3>/experiment_N/`` with ``train.log``, scalar curves,
overlays, and ``REPORT.md`` rollups.

Dataset layout (to be filled when you add the indoor detection set):
  data/<dataset_name>/
    (COCO-style or your chosen layout — TBD)

Assignment reminders:
  - 80/10/10 train/val/test split; all classes present and balanced per split.
  - Report validation mAP (and good/bad prediction examples on val).
  - Final deliverable: Google Colab notebook (to be ported after local training).

Integration note (``exp_logging.manager``): the post-run parser looks for a summary line
containing ``[TEST]``, ``loss=``, and ``mIoU=``. For pure detection metrics you can log
validation mAP in the same line by also including ``mIoU=`` as a numeric placeholder, or
extend ``exp_logging/manager.py::_parse_test_results`` in this repo only. Alternatively,
write ``results.json`` under the experiment directory before confirming the report prompt.
"""

from __future__ import annotations

import argparse
import builtins
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "data"
DATASET_NAME = "indoor_detection"  # logical name for work_dirs hierarchy

# Ensure local imports (``import exp_logging``) work when run as ``python train_and_eval.py``.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def setup_logging(work_dir: Path, ensure_dir_fn) -> None:
    ensure_dir_fn(work_dir)
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


def _run_mock_training(work_dir: Path, tracker) -> None:
    """Placeholder epochs so plots and REPORT sections have something to read."""
    logging.info("MOCK: replace this with real train/val loop + COCOeval mAP.")
    for epoch in range(1, 4):
        tracker.log_scalar("train/loss", 1.0 / epoch, step=epoch)
        tracker.log_scalar("val/loss", 1.1 / epoch, step=epoch)
        tracker.log_scalar("val/miou", 0.1 * epoch, step=epoch)  # tag name reused for curves
        logging.info("[epoch %s] stub metrics logged", epoch)
    # Satisfies manager._parse_test_results (segmentation-oriented; extend or duplicate mAP as mIoU for reports).
    logging.info("[TEST] loss=0.25 | mIoU=0.42 | pixAcc=0.88")


def _patch_input_for_mock_pipeline() -> None:
    """Enough scripted answers for ``ask_experiment_level_and_context`` + finalize + questions."""
    replies = iter(
        [
            "4",  # experiment level
            "default",  # lv2
            "MockDetector",  # lv3 (network tag)
            "y",  # keep logs
            "Smoke test for repo wiring.",
            "Nothing yet; baseline scaffold.",
            "None.",
            "N/A — stub only.",
        ]
    )

    def _fake_input(prompt: str = "") -> str:  # noqa: ARG001
        try:
            return next(replies)
        except StopIteration:
            return ""

    builtins.input = _fake_input  # type: ignore[assignment]


def main() -> None:
    parser = argparse.ArgumentParser(description="DocuSketch indoor detection — train & eval")
    parser.add_argument(
        "--mock-complete",
        action="store_true",
        help="Non-interactive smoke run: creates a real experiment folder under work_dirs/.",
    )
    args = parser.parse_args()

    if args.mock_complete:
        _patch_input_for_mock_pipeline()

    # Non-interactive backends: import matplotlib only after optional input() patch.
    os.environ.setdefault("MPLBACKEND", "Agg")

    from exp_logging.io_utils import ensure_dir
    from exp_logging.manager import (
        ask_experiment_level_and_context,
        begin_experiment_session,
        finalize_experiment_session,
        register_sigint_handler,
    )
    from exp_logging.tracker import ScalarTracker

    meta = ask_experiment_level_and_context(
        dataset_name=DATASET_NAME,
        default_lv3_model_tag="FasterRCNN_ResNet50_FPN",
    )
    session = begin_experiment_session(meta)
    register_sigint_handler(session)
    work_dir = session.temp_dir
    setup_logging(work_dir, ensure_dir)
    tracker = ScalarTracker(work_dir)

    logging.info("DATA_ROOT=%s (place dataset here)", DATA_ROOT)

    try:
        _run_mock_training(work_dir, tracker)
    except KeyboardInterrupt:
        logging.info("Interrupted before finalize; SIGINT handler will offer to keep logs.")
        raise
    finalize_experiment_session(session)


if __name__ == "__main__":
    main()
