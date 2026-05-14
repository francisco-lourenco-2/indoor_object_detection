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

def _safe_num(x, digits=3):
    try:
        v = float(x)
        if math.isfinite(v):
            return f"{v:.{digits}f}"
    except Exception:
        pass
    return "?"

def _find_siblings_best(lv3_dir: Path, current: Path):
    best_dir = None
    best_score = None
    for d in sorted(lv3_dir.glob("experiment_*")):
        r = read_json(d / "results.json", {})
        sc = (r.get("test") or {}).get("mIoU")
        try:
            sc = float(sc)
            if not math.isfinite(sc):
                continue
        except Exception:
            continue
        if best_score is None or sc > best_score:
            best_dir, best_score = d, sc
    return best_dir, best_score

def build_lv4_report(work_dir: Path):
    meta = read_json(work_dir/"meta.json", {})
    scalars = work_dir/"metrics/scalars.jsonl"
    plots_dir = work_dir/"plots"
    plot_curves(scalars, plots_dir)

    results = read_json(work_dir/"results.json", {})
    answers = read_json(work_dir/"answers.json", {})

    # comparison vs best sibling:
    best_dir, best_score = _find_siblings_best(work_dir.parent, work_dir)
    cur = (results.get("test") or {}).get("mIoU")
    arrow, word = "—", "Same"
    try:
        curf = float(cur); bestf = float(best_score) if best_score is not None else None
        if best_dir and best_dir != work_dir and bestf is not None and math.isfinite(curf) and math.isfinite(bestf):
            if curf > bestf + 1e-6: arrow, word = "▲","Improved"
            elif curf < bestf - 1e-6: arrow, word = "▼","Worse"
    except Exception:
        pass

    summary, conclusion = summarize_and_conclude(meta, results, answers)

    # write Markdown
    md = []
    md.append(f"# {meta.get('dataset','?')} / {meta.get('lv2_name','-')} / {meta.get('lv3_name','?')} / {meta.get('lv4_name','?')}")
    md.append("\n## What & Why\n" + (summary or ""))

    md.append("\n## Training curves\n")
    for fn in ["loss_curve.png","miou_curve.png","lr_curve.png"]:
        p = (plots_dir/fn)
        if p.exists(): md.append(f"![{fn}](plots/{fn})")

    md.append("\n## Overlays\n")
    for split in ["overlays_train","overlays_test"]:
        d = work_dir/split
        imgs = sorted([p.name for p in d.glob("*.png")])[:12]
        if imgs:
            md.append(f"\n**{split.replace('_',' ').title()}**")
            for n in imgs:
                md.append(f"![]({split}/{n})")

    # Test results table (sanitized)
    t = results.get("test", {}) or {}
    row = " | ".join([f"mIoU: {_safe_num(t.get('mIoU'))}",
                      f"pixAcc: {_safe_num(t.get('pixel_acc'))}",
                      f"loss: {_safe_num(t.get('loss'))}"])
    md.append("\n## Test results\n" + row)

    best_disp = _safe_num(best_score) if best_score is not None else "N/A"
    md.append(f"\n\n**Comparison vs best sibling:** {arrow} *{word}* (best mIoU={best_disp})")
    md.append("\n\n## Conclusion\n" + (conclusion or ""))
    (work_dir/"REPORT.md").write_text("\n".join(md))

def _collect_lv4_rows(root: Path):
    rows=[]
    for exp in sorted(root.glob("experiment_*")):
        r = read_json(exp/"results.json",{}) or {}
        t = r.get("test",{}) or {}
        rows.append((exp.name, t.get("mIoU"), t.get("pixel_acc"), t.get("loss")))
    return rows

def _collect_lv4_full(root: Path, lv2_name: str, lv3_name: str):
    """Return richer rows incl. answers for L2/L3/L1 summarizers."""
    rows=[]
    for exp in sorted(root.glob("experiment_*")):
        r = read_json(exp/"results.json",{}) or {}
        t = r.get("test",{}) or {}
        a = read_json(exp/"answers.json",{}) or {}
        rows.append({
            "name": exp.name,
            "path": str(exp),
            "lv2": lv2_name,
            "lv3": lv3_name,
            "mIoU": t.get("mIoU"),
            "pixAcc": t.get("pixel_acc"),
            "loss": t.get("loss"),
            "answers": a
        })
    return rows

def build_rollups(work_dir: Path):
    # ================= Level 3 =================
    lv3_dir = work_dir.parent
    lv2_name = lv3_dir.parent.name
    lv3_name = lv3_dir.name
    rows = _collect_lv4_rows(lv3_dir)
    if rows:
        # Find best finite mIoU
        finite = []
        for _, miou, *_ in rows:
            try:
                v = float(miou)
                if math.isfinite(v): finite.append(v)
            except Exception:
                pass
        best = max(finite) if finite else None

        # Table
        lines = ["# " + lv3_dir.name, "",
                 "## Results (Level 4 under this network)",
                 "| Experiment | mIoU | pixAcc | loss |", "|---|---:|---:|---:|"]
        for name, miou, pa, loss in rows:
            m = _safe_num(miou)
            s = f"**{m}**" if (best is not None and m == f"{best:.3f}") else m
            lines.append(f"| {name} | {s} | {_safe_num(pa)} | {_safe_num(loss)} |")

        # Narrative (overview + conclusion)
        meta_lv3 = {"dataset": lv3_dir.parent.parent.name, "lv2_name": lv2_name, "lv3_name": lv3_name}
        full_exps = _collect_lv4_full(lv3_dir, lv2_name, lv3_name)
        best_row = None
        if best is not None:
            for r in full_exps:
                try:
                    if math.isfinite(float(r.get("mIoU"))) and float(r.get("mIoU")) == best:
                        best_row = {"name": r["name"], "path": r["path"], "mIoU": float(r["mIoU"])}
                        break
                except Exception:
                    pass
        ov, co = summarize_lv3(meta_lv3, full_exps, best_row or {})
        lines += ["", "## Overview", (ov or ""), "", "## Conclusion", (co or "")]
        (lv3_dir/"REPORT.md").write_text("\n".join(lines))

    # ================= Level 2 =================
    lv2_dir = lv3_dir.parent
    lv3s = [p for p in lv2_dir.iterdir() if p.is_dir()]
    lines = ["# " + lv2_dir.name, "",
             "## Best per Network (Level 3)",
             "| Network (Lv3) | Best mIoU |", "|---|---:|"]
    best2=None; best2_row=None
    lv3_summaries=[]
    for l3 in lv3s:
        rs = _collect_lv4_rows(l3)
        finite = []
        for _, miou, *_ in rs:
            try:
                v = float(miou)
                if math.isfinite(v): finite.append(v)
            except Exception:
                pass
        if not finite:
            continue
        b = max(finite)
        lines.append(f"| {l3.name} | {b:.3f} |")
        if best2 is None or b>best2:
            best2, best2_row = b, {"lv3_name": l3.name, "best_mIoU": b}
        lv3_summaries.append({"lv3_name": l3.name, "best_mIoU": b,
                              "experiments": _collect_lv4_full(l3, lv2_dir.name, l3.name)})

    meta_lv2 = {"dataset": lv2_dir.parent.name, "lv2_name": lv2_dir.name}
    ov2, co2 = summarize_lv2(meta_lv2, lv3_summaries, best2_row or {})
    lines += ["", "## Overview", (ov2 or ""), "", "## Conclusion", (co2 or "")]
    (lv2_dir/"REPORT.md").write_text("\n".join(lines))

    # ================= Level 1 =================
    lv1_dir = lv2_dir.parent
    lines = ["# " + lv1_dir.name, "",
             "## All Experiments on this Dataset",
             "| Data Mode | Network | Experiment | mIoU |", "|---|---|---|---:|"]
    best1=None; best_row=None
    all_exps=[]
    for l2 in [p for p in lv1_dir.iterdir() if p.is_dir()]:
        for l3 in [p for p in l2.iterdir() if p.is_dir()]:
            full = _collect_lv4_full(l3, l2.name, l3.name)
            for r in full:
                m = r.get("mIoU")
                lines.append(f"| {l2.name} | {l3.name} | {r['name']} | {_safe_num(m)} |")
                all_exps.append(r)
                try:
                    mv = float(m)
                    if math.isfinite(mv) and (best1 is None or mv>best1):
                        best1, best_row = mv, {"lv2": l2.name, "lv3": l3.name, "name": r["name"], "path": r["path"], "mIoU": mv}
                except Exception:
                    pass

    meta_lv1 = {"dataset": lv1_dir.name}
    ov1, co1 = summarize_lv1(meta_lv1, all_exps, best_row or {})
    if best_row is not None:
        lines += [f"\n**Best overall**: ({best_row['lv2']}, {best_row['lv3']}, {best_row['name']}) with **mIoU={best1:.3f}**"]
    lines += ["", "## Overview", (ov1 or ""), "", "## Conclusion", (co1 or "")]
    (lv1_dir/"REPORT.md").write_text("\n".join(lines))
