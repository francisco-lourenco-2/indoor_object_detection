from pathlib import Path
import math
from exp_logging.io_utils import read_json, write_json, ensure_dir
from exp_logging.plots import plot_curves
from exp_logging.openai_helper import (
    summarize_and_conclude,
    summarize_lv3,
    summarize_lv2,
    summarize_lv1,
)

# Object-detection test metrics shown in reports (order preserved).
DETECTION_TEST_KEYS = ("mAP", "AP50", "AP75", "precision", "recall")

_COMPARE_EPS = 1e-6


def _safe_num(x, digits=3):
    try:
        v = float(x)
        if math.isfinite(v):
            return f"{v:.{digits}f}"
    except Exception:
        pass
    return "?"


def _primary_map(test: dict | None) -> float:
    """Ranking / comparison metric: test mAP (falls back to legacy mIoU alias)."""
    test = test or {}
    for key in ("mAP", "mIoU"):
        try:
            v = float(test.get(key))
            if math.isfinite(v):
                return v
        except (TypeError, ValueError):
            pass
    return float("nan")


def _find_best_other_sibling(lv3_dir: Path, current: Path) -> float | None:
    """Best test mAP among sibling experiments, excluding ``current``."""
    best_score = None
    for d in sorted(lv3_dir.glob("experiment_*")):
        if not d.is_dir() or d.name.endswith(".__tmp__"):
            continue
        if d.resolve() == current.resolve():
            continue
        sc = _primary_map(read_json(d / "results.json", {}).get("test"))
        if not math.isfinite(sc):
            continue
        if best_score is None or sc > best_score:
            best_score = sc
    return best_score


def _sibling_comparison_line(work_dir: Path, results: dict) -> str:
    """Human-readable comparison vs the best other experiment under the same Lv3 network."""
    test = results.get("test") or {}
    cur = _primary_map(test)
    lv3_dir = work_dir.parent

    siblings = [
        d for d in lv3_dir.glob("experiment_*") if d.is_dir() and not d.name.endswith(".__tmp__")
    ]
    if len(siblings) <= 1:
        return f"**Comparison vs best sibling:** — *Only experiment* (mAP={_safe_num(cur)})"

    if not math.isfinite(cur):
        return "**Comparison vs best sibling:** — *Unknown* (missing test mAP)"

    best_other = _find_best_other_sibling(lv3_dir, work_dir)
    cur_disp = _safe_num(cur)
    best_disp = _safe_num(best_other) if best_other is not None else "N/A"

    if best_other is None or not math.isfinite(best_other):
        return f"**Comparison vs best sibling:** — *No sibling results* (this run mAP={cur_disp})"

    if cur > best_other + _COMPARE_EPS:
        arrow, word = "▲", "Improved"
    elif cur < best_other - _COMPARE_EPS:
        arrow, word = "▼", "Worse"
    else:
        arrow, word = "—", "Tied"

    return (
        f"**Comparison vs best sibling:** {arrow} *{word}* "
        f"(sibling best mAP={best_disp}, this run mAP={cur_disp})"
    )


def _format_test_results_block(test: dict | None) -> str:
    test = test or {}
    lines = [f"{key}: {_safe_num(test.get(key))}" for key in DETECTION_TEST_KEYS]
    return "\n".join(lines)


def _answers_empty(answers: dict | None) -> bool:
    if not answers:
        return True
    return not any(str(v).strip() for v in answers.values())


def _meta_config_block(meta: dict) -> str:
    """Key training / run settings (values only)."""
    keys = (
        "dataset",
        "lv2_name",
        "lv3_name",
        "lv4_name",
        "model",
        "epochs",
        "imgsz",
        "batch",
        "device",
        "seed",
        "data_yaml",
        "framework",
    )
    lines: list[str] = []
    for name in keys:
        if meta.get(name) is not None:
            lines.append(f"{name}: {meta[name]}")
    aug = meta.get("augment")
    if isinstance(aug, dict) and aug:
        for name, value in aug.items():
            lines.append(f"augment.{name}: {value}")
    return "\n".join(lines)


def build_lv4_report(work_dir: Path):
    meta = read_json(work_dir / "meta.json", {})
    scalars = work_dir / "metrics/scalars.jsonl"
    plots_dir = work_dir / "plots"
    plot_curves(scalars, plots_dir)

    results = read_json(work_dir / "results.json", {})
    answers = read_json(work_dir / "answers.json", {})
    metrics_only = _answers_empty(answers)

    md = []
    md.append(
        f"# {meta.get('dataset', '?')} / {meta.get('lv2_name', '-')} / "
        f"{meta.get('lv3_name', '?')} / {meta.get('lv4_name', '?')}"
    )
    if metrics_only:
        md.append("\n## Configuration\n" + _meta_config_block(meta))
    else:
        summary, conclusion = summarize_and_conclude(meta, results, answers)
        md.append("\n## What & Why\n" + (summary or ""))

    md.append("\n## Training curves\n")
    for fn in ("loss_curve.png", "map_curve.png", "miou_curve.png", "lr_curve.png"):
        p = plots_dir / fn
        if p.exists():
            md.append(f"![{fn}](plots/{fn})")

    md.append("\n## Overlays\n")
    for split in ("overlays_train", "overlays_val", "overlays_test"):
        d = work_dir / split
        imgs = sorted(p.name for p in d.glob("*.png"))[:12]
        if imgs:
            md.append(f"\n**{split.replace('_', ' ').title()}**")
            for n in imgs:
                md.append(f"![]({split}/{n})")

    md.append("\n## Test results\n" + _format_test_results_block(results.get("test")))
    md.append("\n\n" + _sibling_comparison_line(work_dir, results))
    if not metrics_only:
        md.append("\n\n## Conclusion\n" + (conclusion or ""))
    (work_dir / "REPORT.md").write_text("\n".join(md))


def _collect_lv4_rows(root: Path) -> list[tuple]:
    rows = []
    for exp in sorted(root.glob("experiment_*")):
        if not exp.is_dir() or exp.name.endswith(".__tmp__"):
            continue
        t = read_json(exp / "results.json", {}) or {}
        test = t.get("test") or {}
        rows.append(
            (
                exp.name,
                test.get("mAP"),
                test.get("AP50"),
                test.get("AP75"),
                test.get("precision"),
                test.get("recall"),
            )
        )
    return rows


def _collect_lv4_full(root: Path, lv2_name: str, lv3_name: str) -> list[dict]:
    rows = []
    for exp in sorted(root.glob("experiment_*")):
        if not exp.is_dir() or exp.name.endswith(".__tmp__"):
            continue
        t = read_json(exp / "results.json", {}) or {}
        test = t.get("test") or {}
        a = read_json(exp / "answers.json", {}) or {}
        mAP = _primary_map(test)
        rows.append(
            {
                "name": exp.name,
                "path": str(exp),
                "lv2": lv2_name,
                "lv3": lv3_name,
                "mAP": mAP if math.isfinite(mAP) else test.get("mAP"),
                "AP50": test.get("AP50"),
                "AP75": test.get("AP75"),
                "precision": test.get("precision"),
                "recall": test.get("recall"),
                "answers": a,
            }
        )
    return rows


def _best_finite_map(values) -> float | None:
    finite = []
    for v in values:
        try:
            f = float(v)
            if math.isfinite(f):
                finite.append(f)
        except (TypeError, ValueError):
            pass
    return max(finite) if finite else None


def build_rollups(work_dir: Path):
    # ================= Level 3 =================
    lv3_dir = work_dir.parent
    lv2_name = lv3_dir.parent.name
    lv3_name = lv3_dir.name
    rows = _collect_lv4_rows(lv3_dir)
    if rows:
        best = _best_finite_map(row[1] for row in rows)

        lines = [
            "# " + lv3_dir.name,
            "",
            "## Results (Level 4 under this network)",
            "| Experiment | mAP | AP50 | AP75 | precision | recall |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, mAP, ap50, ap75, prec, rec in rows:
            m = _safe_num(mAP)
            s = f"**{m}**" if (best is not None and m == f"{best:.3f}") else m
            lines.append(
                f"| {name} | {s} | {_safe_num(ap50)} | {_safe_num(ap75)} | "
                f"{_safe_num(prec)} | {_safe_num(rec)} |"
            )

        meta_lv3 = {"dataset": lv3_dir.parent.parent.name, "lv2_name": lv2_name, "lv3_name": lv3_name}
        full_exps = _collect_lv4_full(lv3_dir, lv2_name, lv3_name)
        best_row = None
        if best is not None:
            for r in full_exps:
                try:
                    if math.isfinite(float(r.get("mAP"))) and float(r.get("mAP")) == best:
                        best_row = {"name": r["name"], "path": r["path"], "mAP": float(r["mAP"])}
                        break
                except (TypeError, ValueError):
                    pass
        ov, co = summarize_lv3(meta_lv3, full_exps, best_row or {})
        lines += ["", "## Overview", (ov or ""), "", "## Conclusion", (co or "")]
        (lv3_dir / "REPORT.md").write_text("\n".join(lines))

    # ================= Level 2 =================
    lv2_dir = lv3_dir.parent
    lv3s = [p for p in lv2_dir.iterdir() if p.is_dir()]
    lines = [
        "# " + lv2_dir.name,
        "",
        "## Best per Network (Level 3)",
        "| Network (Lv3) | Best mAP |",
        "|---|---:|",
    ]
    best2 = None
    best2_row = None
    lv3_summaries = []
    for l3 in lv3s:
        rs = _collect_lv4_rows(l3)
        maps = [r[1] for r in rs]
        b = _best_finite_map(maps)
        if b is None:
            continue
        lines.append(f"| {l3.name} | {b:.3f} |")
        if best2 is None or b > best2:
            best2, best2_row = b, {"lv3_name": l3.name, "best_mAP": b}
        lv3_summaries.append(
            {
                "lv3_name": l3.name,
                "best_mAP": b,
                "experiments": _collect_lv4_full(l3, lv2_dir.name, l3.name),
            }
        )

    meta_lv2 = {"dataset": lv2_dir.parent.name, "lv2_name": lv2_dir.name}
    ov2, co2 = summarize_lv2(meta_lv2, lv3_summaries, best2_row or {})
    lines += ["", "## Overview", (ov2 or ""), "", "## Conclusion", (co2 or "")]
    (lv2_dir / "REPORT.md").write_text("\n".join(lines))

    # ================= Level 1 =================
    lv1_dir = lv2_dir.parent
    lines = [
        "# " + lv1_dir.name,
        "",
        "## All Experiments on this Dataset",
        "| Data Mode | Network | Experiment | mAP |",
        "|---|---|---|---:|",
    ]
    best1 = None
    best_row = None
    all_exps = []
    for l2 in [p for p in lv1_dir.iterdir() if p.is_dir()]:
        for l3 in [p for p in l2.iterdir() if p.is_dir()]:
            full = _collect_lv4_full(l3, l2.name, l3.name)
            for r in full:
                m = r.get("mAP")
                lines.append(f"| {l2.name} | {l3.name} | {r['name']} | {_safe_num(m)} |")
                all_exps.append(r)
                try:
                    mv = float(m)
                    if math.isfinite(mv) and (best1 is None or mv > best1):
                        best1 = mv
                        best_row = {
                            "lv2": l2.name,
                            "lv3": l3.name,
                            "name": r["name"],
                            "path": r["path"],
                            "mAP": mv,
                        }
                except (TypeError, ValueError):
                    pass

    meta_lv1 = {"dataset": lv1_dir.name}
    ov1, co1 = summarize_lv1(meta_lv1, all_exps, best_row or {})
    if best_row is not None:
        lines += [
            f"\n**Best overall**: ({best_row['lv2']}, {best_row['lv3']}, {best_row['name']}) "
            f"with **mAP={best1:.3f}**"
        ]
    lines += ["", "## Overview", (ov1 or ""), "", "## Conclusion", (co1 or "")]
    (lv1_dir / "REPORT.md").write_text("\n".join(lines))


def rebuild_reports_under(work_dirs_dataset: Path, *, refresh_lv4_plots: bool = True) -> None:
    """
    Rebuild Lv1–Lv3 rollups from existing ``results.json`` files.

    Optionally regenerate training plots (``map_curve.png``, etc.) for each Lv4 experiment
    without rewriting ``REPORT.md`` (preserves hand-edited Lv4 narratives).
    """
    work_dirs_dataset = work_dirs_dataset.resolve()
    lv3_dirs: set[Path] = set()
    for results_path in work_dirs_dataset.rglob("results.json"):
        exp_dir = results_path.parent
        if not exp_dir.name.startswith("experiment_"):
            continue
        if refresh_lv4_plots and (exp_dir / "metrics/scalars.jsonl").exists():
            plot_curves(exp_dir / "metrics/scalars.jsonl", exp_dir / "plots")
        lv3_dirs.add(exp_dir.parent)

    for lv3 in sorted(lv3_dirs):
        latest = max(
            (d for d in lv3.glob("experiment_*") if d.is_dir() and not d.name.endswith(".__tmp__")),
            key=lambda p: p.name,
            default=None,
        )
        if latest is not None:
            build_rollups(latest)
