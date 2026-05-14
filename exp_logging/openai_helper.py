# exp_logging/openai_helper.py
import os
import json
import re
import textwrap
from typing import Tuple

try:
    # Official SDK
    from openai import OpenAI, APIError, RateLimitError, APITimeoutError
except Exception:
    OpenAI = None
    APIError = RateLimitError = APITimeoutError = Exception  # type: ignore


FALLBACK_THRESHOLD_MIOU = 0.50  # tweak to your bar for "adopt"


def _fallback(meta: dict, metrics: dict, answers: dict) -> Tuple[str, str]:
    """Minimal, deterministic fallback when API key is missing or API fails."""
    goal = next(iter(answers.values()), "").strip()
    test = metrics.get("test", {}) or {}
    miou = test.get("mIoU", "?")
    try:
        miou_f = float(miou)
        miou_str = f"{miou_f:.3f}"
    except Exception:
        miou_f = None
        miou_str = str(miou)

    changed = answers.get("What changed compared to the baseline/previous best?", "") or "N/A"
    summary = f"{meta.get('lv4_name','experiment')}: {goal}\nChanges: {changed}"

    rec = "keep iterating"
    if isinstance(miou_f, float):
        rec = "adopt" if miou_f >= FALLBACK_THRESHOLD_MIOU else "keep iterating"

    conclusion = f"Test mIoU={miou_str}. Recommended: {rec}."
    return summary, conclusion


def _load_optional_style_prompt() -> str:
    """
    If you keep a small style guide snippet (markdown/text) next to your reports,
    you can point this to it (e.g., exp_logging/style_prompt.txt). Keep it short.
    """
    for candidate in (
        os.getenv("EXP_LOGGING_STYLE_PROMPT"),
        "exp_logging/style_prompt.txt",
        "style_prompt.txt",
    ):
        if candidate and os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                pass
    return ""


def _defence_json(txt: str) -> str:
    """Strip common code fences or leading/trailing fluff before json.loads."""
    if not txt:
        return txt
    # Remove markdown code fences if present
    txt = re.sub(r"^\s*```(?:json)?\s*", "", txt.strip(), flags=re.IGNORECASE)
    txt = re.sub(r"\s*```\s*$", "", txt.strip())
    # Trim any leading/trailing non-json junk
    first = txt.find("{")
    last = txt.rfind("}")
    if first != -1 and last != -1 and last > first:
        return txt[first:last+1]
    return txt


def _responses_json(system_prompt: str, user_json_payload: dict, keys=("overview","conclusion")):
    """Shared tiny wrapper; falls back to compact deterministic text on errors/missing key."""
    if not os.getenv("OPENAI_API_KEY") or OpenAI is None:
        # ultra-compact deterministic fallback
        exps = user_json_payload.get("experiments", [])
        best = user_json_payload.get("best", {})
        ov = f"{len(exps)} experiments. Best mIoU={best.get('mIoU','?')} ({best.get('name','?')})."
        co = f"Use {best.get('path','?')} in production." if isinstance(best.get("mIoU",None),(int,float)) else "Keep iterating."
        return ov, co

    client = OpenAI()
    try:
        resp = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            input=[
                {"role":"system","content":[{"type":"input_text","text":system_prompt}]},
                {"role":"user","content":[{"type":"input_text","text":json.dumps(user_json_payload, ensure_ascii=False)}]},
            ],
            temperature=0.2,
            max_output_tokens=600,
        )
        text = _defence_json(getattr(resp, "output_text", "") or "")
        data = json.loads(text)
        return (str(data.get(keys[0],"")).strip() or "",
                str(data.get(keys[1],"")).strip() or "")
    except Exception:
        # same deterministic fallback
        exps = user_json_payload.get("experiments", [])
        best = user_json_payload.get("best", {})
        ov = f"{len(exps)} experiments. Best mIoU={best.get('mIoU','?')} ({best.get('name','?')})."
        co = f"Use {best.get('path','?')} in production." if isinstance(best.get("mIoU",None),(int,float)) else "Keep iterating."
        return ov, co


def summarize_lv3(meta_lv3: dict, experiments: list, best: dict):
    """
    meta_lv3: {dataset, lv2_name, lv3_name}
    experiments: [{"name","path","mIoU","pixAcc","loss","answers"}]
    best: {"name","path","mIoU"}
    """
    system = (
        "You write concise, presentation-ready rollups for model-level (Level 3) results in a roof-segmentation project.\n"
        "Output STRICT JSON with keys: overview, conclusion. No markdown, no extra keys.\n"
        "overview: 2–5 sentences describing WHAT model was explored and WHY (infer from answers),\n"
        "          plus a short synthesis of results across Level-4 runs focusing on test mIoU.\n"
        "conclusion: 1–3 sentences naming the best L4 experiment and whether to continue or adopt."
    )
    payload = {"level":3, "meta":meta_lv3, "experiments":experiments, "best":best}
    return _responses_json(system, payload)


def summarize_lv2(meta_lv2: dict, lv3_summaries: list, best: dict):
    """
    meta_lv2: {dataset, lv2_name}
    lv3_summaries: [{"lv3_name","best_mIoU","experiments":[...]}]
    best: {"lv3_name","best_mIoU"}
    """
    system = (
        "You write concise, presentation-ready rollups for data-level (Level 2) results.\n"
        "Output STRICT JSON with keys: overview, conclusion.\n"
        "overview: 2–5 sentences explaining WHAT data-related change was done and WHY (use per-experiment answers),\n"
        "          then compare networks by their best test mIoU.\n"
        "conclusion: 1–3 sentences naming the best-performing network for this data setting and next steps."
    )
    payload = {"level":2, "meta":meta_lv2, "networks":lv3_summaries, "best":best}
    return _responses_json(system, payload)


def summarize_lv1(meta_lv1: dict, all_experiments: list, best: dict):
    """
    meta_lv1: {dataset}
    all_experiments: [{"lv2","lv3","name","path","mIoU","answers"}]  # across whole dataset
    best: {"lv2","lv3","name","path","mIoU"}
    """
    system = (
        "You write concise, presentation-ready dataset-level (Level 1) summaries.\n"
        "Output STRICT JSON with keys: overview, conclusion.\n"
        "overview: 3–6 sentences giving a high-level narrative of goals/challenges on this dataset\n"
        "          and a brief review of experiments across data modes and networks; focus on test mIoU only.\n"
        "conclusion: 1–3 sentences naming the best solution and recommending production usage."
    )
    payload = {"level":1, "meta":meta_lv1, "experiments":all_experiments, "best":best}
    return _responses_json(system, payload)


def summarize_and_conclude(meta: dict, metrics: dict, answers: dict) -> Tuple[str, str]:
    """
    Returns (summary, conclusion) strings.
    Uses OpenAI Responses API when OPENAI_API_KEY is set; otherwise falls back.
    """
    if not os.getenv("OPENAI_API_KEY") or OpenAI is None:
        return _fallback(meta, metrics, answers)

    # Build the input we’ll feed the model
    style = _load_optional_style_prompt()
    style_snippet = f"\nStyle notes:\n{style}" if style else ""

    # Keep inputs compact and unambiguous for deterministic outputs
    test = metrics.get("test", {}) or {}
    # Make a lean JSON snapshot we can show to the model
    model_context = {
        "levels": {
            "level": meta.get("level"),
            "dataset": meta.get("dataset"),
            "lv2_name": meta.get("lv2_name"),
            "lv3_name": meta.get("lv3_name"),
            "lv4_name": meta.get("lv4_name"),
        },
        "training": {
            "encoder_name": meta.get("encoder_name"),
            "in_channels": meta.get("in_channels"),
            "num_classes": meta.get("num_classes"),
            "batch_size": meta.get("batch_size"),
            "lr": meta.get("lr"),
            "epochs": meta.get("epochs"),
            "optimizer": meta.get("optimizer"),
            "scheduler": meta.get("scheduler"),
        },
        "results": {
            "test": test  # expected keys: loss, mIoU, pixel_acc, maybe per-class IoUs elsewhere
        },
        "answers": answers,  # your 3–5 answers captured after training
    }

    # System instructions: short, non-fluffy; professional tone
    system_instructions = textwrap.dedent(
        "You are an assistant that writes concise, professional experiment summaries for a\n"
        "computer-vision segmentation project. Your writing should be:\n"
        "- short, precise, and presentation-ready,\n"
        "- non-fluffy, with concrete statements,\n"
        "- easy to read for an engineering audience,\n"
        "- consistent in tone across runs.\n\n"
        "Output STRICT JSON with exactly two keys:\n"
        '  "summary": one short paragraph (2–5 sentences) stating WHAT changed and WHY, and any key insights.\n'
        '  "conclusion": one short paragraph (1–3 sentences) with an actionable verdict (e.g., adopt / keep iterating) based on the test results.\n'
        "Do NOT include markdown or extra keys. No code fences.\n"
    ).strip() + style_snippet

    user_prompt = textwrap.dedent(
        "Context (compact JSON):\n"
        f"{json.dumps(model_context, ensure_ascii=False)}\n\n"
        "Requirements:\n"
        "- Keep it concise.\n"
        "- Avoid speculation: if something is unknown, omit it.\n"
        "- Base the verdict primarily on test mIoU; mention it explicitly.\n"
        "- If improvements depend on data or setup changes, say it plainly (short).\n"
        '- Output JSON ONLY with keys: "summary", "conclusion".\n'
    ).strip()

    # Call the Responses API (official SDK)
    client = OpenAI()
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
    try:
        resp = client.responses.create(
            model=model_name,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_instructions}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
            temperature=0.2,
            max_output_tokens=500,
        )
        text = getattr(resp, "output_text", None) or ""

        # Expect strict JSON with "summary" and "conclusion"
        text = _defence_json(text)
        data = json.loads(text)
        summary = str(data.get("summary", "")).strip()
        conclusion = str(data.get("conclusion", "")).strip()

        # If the model was too chatty or missed keys, fallback gracefully
        if not summary or not conclusion:
            return _fallback(meta, metrics, answers)
        return summary, conclusion

    except (APITimeoutError, RateLimitError, APIError, json.JSONDecodeError, Exception):
        # Any error ⇒ non-fatal fallback
        return _fallback(meta, metrics, answers)
