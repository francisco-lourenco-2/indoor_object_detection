from pathlib import Path
import json
import matplotlib.pyplot as plt


def _load_series(scalars_jsonl: Path):
    if not scalars_jsonl.exists():
        return []
    lines = [line for line in scalars_jsonl.read_text().splitlines() if line.strip()]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def _collect(series, tag):
    pts = [(r.get("step"), r.get("value")) for r in series if r.get("tag") == tag]
    pts = [(int(s), float(v)) for s, v in pts if s is not None and v is not None]
    return sorted(pts, key=lambda x: x[0])


def _plot_xy(xs, ys, label, xlabel, ylabel, title):
    plt.plot(xs, ys, marker="o", linestyle="-", label=label, markersize=3.0, linewidth=1.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linewidth=0.3, alpha=0.6)


def plot_curves(scalars_jsonl: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    series = _load_series(scalars_jsonl)

    train_loss = _collect(series, "train/loss")
    val_loss = _collect(series, "val/loss")
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

    map_pts = _collect(series, "val/mAP") or _collect(series, "val/miou")
    if map_pts:
        xs, ys = zip(*map_pts)
        plt.figure()
        _plot_xy(xs, ys, "val mAP", "epoch", "mAP", "Validation mAP")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "map_curve.png", dpi=160)
        plt.close()

    lrs = _collect(series, "opt/lr")
    if lrs:
        xs, ys = zip(*lrs)
        xlab = "epoch" if xs and max(xs) <= 2000 else "step"
        plt.figure()
        _plot_xy(xs, ys, "lr", xlab, "lr", "Learning Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "lr_curve.png", dpi=160)
        plt.close()
