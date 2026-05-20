from dataclasses import dataclass
from pathlib import Path
import json, os, sys, signal, time
from exp_logging.io_utils import ensure_dir, write_json, read_json, rm_tree
from exp_logging.report_builder import build_lv4_report, build_rollups
from exp_logging.questions import ask as ask_questions
import math

ROOT = Path("work_dirs")

@dataclass
class Session:
    meta:dict
    work_dir:Path
    temp_dir:Path
    active:bool=True

def _auto_lv4_name(lv3_dir: Path) -> str:
    n = 1
    while (lv3_dir / f"experiment_{n}").exists() or (lv3_dir / f"experiment_{n}.__tmp__").exists():
        n += 1
    return f"experiment_{n}"


def ask_experiment_level_and_context(dataset_name:str, default_lv3_model_tag:str)->dict:
    print("\nExperiment Level? [1: dataset | 2: data | 3: network | 4: minor]")
    while True:
        try:
            lvl = int(input("> ").strip())
            if lvl in (1,2,3,4): break
        except: pass
        print("Choose 1/2/3/4")
    lv1 = dataset_name
    lv2 = input("Lv2 name (data change) [if level=2, required; else optional]: ").strip()
    lv3 = input(f"Lv3 name (network tag) [default {default_lv3_model_tag}]: ").strip() or default_lv3_model_tag
    return {"level":lvl, "dataset":lv1, "lv2_name":lv2 or "default", "lv3_name":lv3}

def begin_experiment_session(meta:dict)->Session:
    lv1 = ROOT / meta["dataset"]
    lv2 = lv1 / meta["lv2_name"]
    lv3 = lv2 / meta["lv3_name"]
    ensure_dir(lv3)
    lv4 = _auto_lv4_name(lv3)
    meta["lv4_name"] = lv4
    final_dir = lv3 / lv4
    tmp_dir = final_dir.with_name(final_dir.name + ".__tmp__")
    ensure_dir(tmp_dir)
    meta["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(meta, tmp_dir/"meta.json")
    return Session(meta=meta, work_dir=final_dir, temp_dir=tmp_dir)

def register_sigint_handler(session:Session):
    def handler(sig, frame):
        print("\n[SIGINT] Training interrupted.")
        _post_training_prompt(session)
    signal.signal(signal.SIGINT, handler)

def _parse_test_results(dir: Path):
    """
    Extract the last summary test line, e.g.
    ``[TEST] mAP=... | AP50=... | AP75=... | precision=... | recall=...``.
    Legacy lines with mIoU/pixAcc/loss are still accepted (mIoU aliases mAP).
    """
    log = dir / "train.log"
    out = {"test": {}}
    if not log.exists():
        return out
    lines = log.read_text().splitlines()
    for line in reversed(lines):
        if "[TEST]" not in line:
            continue
        seg = line.split("[TEST]", 1)[1]
        parts = {}
        for kv in seg.split("|"):
            kv = kv.strip()
            if "=" in kv:
                k, v = kv.split("=", 1)
                parts[k.strip()] = v.strip()

        def _num(key):
            try:
                v = float(parts.get(key, "nan"))
                return v if math.isfinite(v) else float("nan")
            except Exception:
                return float("nan")

        mAP = _num("mAP")
        if not math.isfinite(mAP):
            mAP = _num("mIoU")

        if not math.isfinite(mAP):
            continue

        out["test"] = {
            "mAP": mAP,
            "AP50": _num("AP50"),
            "AP75": _num("AP75"),
            "precision": _num("precision"),
            "recall": _num("recall"),
        }
        break
    return out

def commit_experiment_session(
    session: Session,
    *,
    keep: bool,
    answers: dict | None = None,
    build_reports: bool = True,
) -> Path | None:
    """Move temp run into ``experiment_N/``; optionally write answers and reports."""
    tmp = session.temp_dir
    if not keep:
        rm_tree(tmp)
        print("Discarded.")
        return None

    ensure_dir(session.work_dir.parent)
    dest = session.work_dir
    if dest.exists():
        base = dest.parent
        new_lv4 = _auto_lv4_name(base)
        session.meta["lv4_name"] = new_lv4
        write_json(session.meta, tmp / "meta.json")
        dest = base / new_lv4
        session.work_dir = dest

    os.replace(tmp, dest)

    if answers is None:
        answers = ask_questions(session.meta["level"])
    write_json(answers, session.work_dir / "answers.json")

    existing_results = read_json(session.work_dir / "results.json", {})
    parsed = _parse_test_results(session.work_dir)
    if not existing_results:
        existing_results = parsed
    else:
        parsed_test = parsed.get("test") or {}
        if parsed_test:
            existing_results.setdefault("test", {}).update(parsed_test)
    write_json(existing_results, session.work_dir / "results.json")

    if build_reports:
        import importlib
        import exp_logging.report_builder as _report_builder

        importlib.reload(_report_builder)
        _report_builder.build_lv4_report(session.work_dir)
        _report_builder.build_rollups(session.work_dir)
    print(f"Saved: {session.work_dir}")
    return session.work_dir


def finalize_experiment_session(
    session: Session,
    *,
    keep: bool | None = None,
    answers: dict | None = None,
    build_reports: bool = True,
    exit_on_finish: bool = True,
) -> Path | None:
    """
    Finalize an experiment session after training.

    If ``keep`` is None, prompts on the terminal. Set ``exit_on_finish=False`` when
    calling from a notebook so the kernel is not terminated.
    """
    if keep is None:
        print("\nGenerate report and keep logs? [y/N]")
        keep = input("> ").strip().lower().startswith("y")

    work_dir = commit_experiment_session(
        session, keep=keep, answers=answers, build_reports=build_reports
    )
    if exit_on_finish:
        sys.exit(0)
    return work_dir


def _post_training_prompt(session: Session) -> None:
    finalize_experiment_session(session)
