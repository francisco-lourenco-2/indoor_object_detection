from pathlib import Path
import json
import math
import matplotlib.pyplot as plt

def _load_series(scalars_jsonl: Path):
    if not scalars_jsonl.exists():
        return []
    lines = [l for l in scalars_jsonl.read_text().splitlines() if l.strip()]
    out = []
    for l in lines:
        try:
            out.append(json.loads(l))
        except Exception:
            pass
    return out

def _collect(series, tag):
    pts = [(r.get("step"), r.get("value")) for r in series if r.get("tag")==tag]
    pts = [(int(s), float(v)) for s, v in pts if s is not None and v is not None]
    return sorted(pts, key=lambda x: x[0])

# replace this helper in plots.py
def _plot_xy(xs, ys, label, xlabel, ylabel, title):
    # smaller dots & lines
    plt.plot(
        xs, ys,
        marker="o",
        linestyle="-",
        linewidth=1.0,     # was default; smaller line
        markersize=3.0,    # smaller dots
        markeredgewidth=0  # keep dots clean
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linewidth=0.3, alpha=0.6)  # subtler grid
    # if label:
    #     # keep legend compact everywhere we call _plot_xy
    #     leg = plt.legend(fontsize=8)
    #     if leg:
    #         for lh in leg.legend_handles:
    #             lh.set_linewidth(1.0)


def _plot_xy(xs, ys, label, xlabel, ylabel, title):
    plt.plot(xs, ys, marker="o", linestyle="-", label=label)  # dots + lines
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

def plot_curves(scalars_jsonl: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    series = _load_series(scalars_jsonl)

    # ===== Loss (train + val in one figure) =====
    train_loss = _collect(series, "train/loss")
    val_loss   = _collect(series, "val/loss")
    if train_loss or val_loss:
        plt.figure()
        if train_loss:
            xs, ys = zip(*train_loss)
            _plot_xy(xs, ys, "train loss", "epoch", "loss", "Training / Validation Loss")
        if val_loss:
            xs, ys = zip(*val_loss)
            _plot_xy(xs, ys, "val loss", "epoch", "loss", "Training / Validation Loss")
        if train_loss and val_loss:
            plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "loss_curve.png", dpi=160)
        plt.close()

    # ===== mIoU (validation) =====
    miou = _collect(series, "val/miou")
    if miou:
        xs, ys = zip(*miou)
        plt.figure()
        _plot_xy(xs, ys, "val mIoU", "epoch", "mIoU", "Validation mIoU")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "miou_curve.png", dpi=160)
        plt.close()

    # # ===== LR schedule =====
    # lrs = _collect(series, "opt/lr")
    # if lrs:
    #     xs, ys = zip(*lrs)
    #     # Try to guess epoch vs step: if monotonically decreasing #points ~ epochs, treat as epoch
    #     xlab = "epoch" if xs and (max(xs) <= 2000) else "step"
    #     plt.figure()
    #     _plot_xy(xs, ys, "lr", xlab, "lr", "Learning Rate")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(outdir / "lr_curve.png", dpi=160)
    #     plt.close()
