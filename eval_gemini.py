"""
Gemini evaluation script for the occlusion benchmark.
Mirrors eval_gpt5.py exactly, replacing only the API client and ask_one_question.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


VALID_LETTERS = "ABCDEFGHIJ"
PRED_FIELD = "pred_answer"


# ── Key loading ───────────────────────────────────────────────────────────────

def _read_key_from_file(path: Path) -> str:
    s = path.read_text(encoding="utf-8").strip()
    if "=" in s and s.split("=", 1)[0].strip().isupper():
        s = s.split("=", 1)[1].strip().strip("\"'")
    return s


def load_gemini_api_key() -> str:
    key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if key:
        return key
    script_dir = Path(__file__).resolve().parent
    for d in (script_dir, script_dir.parent):
        p = d / ".gemini_key"
        if p.is_file():
            try:
                k = _read_key_from_file(p).strip()
                if k:
                    return k
            except Exception:
                continue
    raise ValueError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) or create .gemini_key")


# ── JSON helpers ──────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON root: {path}")
    return data


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Prediction JSONL sidecar ──────────────────────────────────────────────────

def _question_stable_id(q: dict[str, Any]) -> str | None:
    rid = q.get("question_id") if q.get("question_id") is not None else q.get("id")
    if rid is None:
        return None
    s = str(rid).strip()
    return s or None


def default_benchmark_preds_jsonl_path(out_json: Path) -> Path:
    return out_json.with_name(out_json.stem + "_preds.jsonl")


def resolve_benchmark_preds_jsonl_path(out_json: Path, explicit: str) -> Path:
    if (explicit or "").strip():
        return Path(explicit).expanduser().resolve()
    return default_benchmark_preds_jsonl_path(out_json)


def append_pred_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_preds_jsonl_map(path: Path) -> dict[tuple[str, str], str]:
    m: dict[tuple[str, str], str] = {}
    if not path.is_file():
        return m
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(obj.get("question_id", "")).strip()
            pf  = str(obj.get("pred_field", "")).strip()
            pred = str(obj.get("pred", "")).strip()
            if qid and pf:
                m[(qid, pf)] = pred
    return m


def merge_preds_map_into_benchmark_work(
    work: dict[str, Any],
    preds: dict[tuple[str, str], str],
    conditions: list[str],
    pred_fields: list[str],
) -> int:
    applied = 0
    if not preds:
        return 0
    for scene in work.get("scenes") or []:
        if not isinstance(scene, dict):
            continue
        for target in scene.get("targets") or []:
            if not isinstance(target, dict):
                continue
            for _c, _u, block in _iter_target_blocks(target, conditions):
                for q in block.get("questions") or []:
                    if not isinstance(q, dict):
                        continue
                    qid = _question_stable_id(q)
                    if not qid:
                        continue
                    for pf in pred_fields:
                        key = (qid, pf)
                        if key in preds:
                            q[pf] = preds[key]
                            applied += 1
    return applied


def export_benchmark_preds_to_jsonl(
    src_json: Path,
    dst_jsonl: Path,
    pred_fields: list[str],
    conditions: list[str],
) -> int:
    data = load_json(src_json)
    if not is_occlusion_benchmark(data):
        return 0
    dst_jsonl.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for scene in data.get("scenes") or []:
        if not isinstance(scene, dict):
            continue
        for target in scene.get("targets") or []:
            if not isinstance(target, dict):
                continue
            for _c, _u, block in _iter_target_blocks(target, conditions):
                for q in block.get("questions") or []:
                    if not isinstance(q, dict):
                        continue
                    qid = _question_stable_id(q)
                    if not qid:
                        continue
                    for pf in pred_fields:
                        if pf not in q:
                            continue
                        letter = normalize_letter(q.get(pf))
                        if letter:
                            lines.append(json.dumps(
                                {"question_id": qid, "pred_field": pf, "pred": letter},
                                ensure_ascii=False,
                            ))
    if lines:
        dst_jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def filter_jsonl_drop_question_ids(path: Path, drop_ids: set[str]) -> int:
    if not path.is_file() or not drop_ids:
        return 0
    kept: list[str] = []
    removed = 0
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                kept.append(raw.rstrip("\n"))
                continue
            qid = str(obj.get("question_id", "")).strip()
            if qid in drop_ids:
                removed += 1
                continue
            kept.append(line)
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return removed


def _parse_question_types_csv(s: str) -> set[str]:
    parts = [p.strip() for p in (s or "").split(",")]
    return {p for p in parts if p}


def _clear_benchmark_predictions_for_types(
    work: dict[str, Any],
    pred_fields: list[str],
    types: set[str],
    conditions: list[str],
) -> tuple[int, set[str]]:
    cleared = 0
    cleared_ids: set[str] = set()
    if not types:
        return 0, cleared_ids
    for scene in work.get("scenes") or []:
        if not isinstance(scene, dict):
            continue
        for target in scene.get("targets") or []:
            if not isinstance(target, dict):
                continue
            for _cond, _uid, block in _iter_target_blocks(target, conditions):
                for q in block.get("questions") or []:
                    if not isinstance(q, dict):
                        continue
                    if str(q.get("question_type", "")) not in types:
                        continue
                    qid = _question_stable_id(q)
                    if qid:
                        cleared_ids.add(qid)
                    for pf in pred_fields:
                        if pf in q:
                            q.pop(pf, None)
                            cleared += 1
    return cleared, cleared_ids


def _count_existing_predictions(payload: dict[str, Any], pred_field: str) -> int:
    if is_occlusion_benchmark(payload):
        n = 0
        for scene in payload.get("scenes") or []:
            if not isinstance(scene, dict):
                continue
            for target in scene.get("targets") or []:
                if not isinstance(target, dict):
                    continue
                for _, __, block in _iter_target_blocks(
                    target, ("clean", "partial_occlusion", "full_occlusion")
                ):
                    for q in block.get("questions") or []:
                        if isinstance(q, dict) and normalize_letter(q.get(pred_field)):
                            n += 1
        return n
    n = 0
    for q in iter_questions(payload):
        if normalize_letter(q.get(pred_field)):
            n += 1
    return n


# ── Path helpers ──────────────────────────────────────────────────────────────

def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in (s or "").strip())


def default_benchmark_output_path(benchmark_json: Path, model: str, modality: str) -> Path:
    model_tok    = _safe_name(model)    or "model"
    modality_tok = _safe_name(modality) or "modality"
    stem = benchmark_json.stem if benchmark_json.suffix else benchmark_json.name
    return benchmark_json.with_name(f"{stem}_eval_{model_tok}_{modality_tok}.json")


def default_benchmark_merged_output_path(benchmark_json: Path, model: str) -> Path:
    model_tok = _safe_name(model) or "model"
    stem = benchmark_json.stem if benchmark_json.suffix else benchmark_json.name
    return benchmark_json.with_name(f"{stem}_eval_{model_tok}.json")


def is_occlusion_benchmark(payload: dict[str, Any]) -> bool:
    scenes = payload.get("scenes")
    return isinstance(scenes, list) and len(scenes) > 0 and isinstance(scenes[0], dict)


def benchmark_path_anchor(payload: dict[str, Any], benchmark_json: Path) -> Path:
    meta = payload.get("meta") or {}
    pa = meta.get("path_anchor")
    if pa is None or pa == "":
        return benchmark_json.parent.resolve()
    if meta.get("path_anchor_relative_to") == "benchmark_json_parent":
        return (benchmark_json.parent / str(pa)).resolve()
    p = Path(str(pa))
    return p.resolve() if p.is_absolute() else (benchmark_json.parent / p).resolve()


def resolve_benchmark_media_path(path_str: str | None, anchor: Path) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute():
        out = p.resolve()
        return out if out.is_file() or out.is_dir() else out
    return (anchor / path_str).resolve()


def collect_multiview_paths(block: dict[str, Any], anchor: Path, max_images: int) -> list[Path]:
    ir = block.get("image_retrieval") or {}
    raw_list = ((ir.get("multiview") or {}).get("image_paths")) or []
    out: list[Path] = []
    for s in raw_list:
        r = resolve_benchmark_media_path(str(s) if s is not None else None, anchor)
        if r is not None and r.is_file():
            out.append(r)
    if out:
        return sorted(set(out), key=lambda x: str(x))[:max_images]
    scene_dir_s = block.get("scene_dir")
    if scene_dir_s:
        sd = resolve_benchmark_media_path(str(scene_dir_s), anchor)
        if sd is not None and sd.is_dir():
            mv = sd / "multiview"
            if mv.is_dir():
                return list_images(mv, max_images)
    return []


def oracle_image_path(block: dict[str, Any], anchor: Path) -> Path | None:
    ir = block.get("image_retrieval") or {}
    of = ir.get("oracle_frame") or {}
    p  = of.get("image_path")
    r  = resolve_benchmark_media_path(str(p) if p else None, anchor)
    if r is not None and r.is_file():
        return r
    return None


def walkthrough_path_for_block(block: dict[str, Any], anchor: Path) -> Path | None:
    scene_dir_s = block.get("scene_dir")
    if not scene_dir_s:
        return None
    sd = resolve_benchmark_media_path(str(scene_dir_s), anchor)
    if sd is None or not sd.is_dir():
        return None
    vid = sd / "walkthrough.mp4"
    return vid if vid.is_file() else None


def _iter_target_blocks(
    target: dict[str, Any],
    conditions: tuple[str, ...] | list[str],
) -> list[tuple[str, str, dict[str, Any]]]:
    results = []
    if "clean" in conditions:
        block = target.get("clean")
        if isinstance(block, dict):
            results.append(("clean", target.get("unit_id") or target.get("target_id", ""), block))
    for occ in target.get("occlusions") or []:
        if not isinstance(occ, dict):
            continue
        occ_type = occ.get("occlusion_type", "")
        if occ_type not in conditions:
            continue
        block = {
            "scene_dir":       occ.get("scene_dir"),
            "image_retrieval": occ.get("image_retrieval"),
            "questions":       occ.get("questions"),
            "scene_type":      occ_type,
        }
        results.append((occ_type, occ.get("unit_id", ""), block))
    return results


# ── Evaluation ────────────────────────────────────────────────────────────────

def _score_block(block: dict[str, Any], pred_field: str) -> dict[str, dict[str, int]]:
    scores: dict[str, dict[str, int]] = {}
    for q in block.get("questions") or []:
        if not isinstance(q, dict):
            continue
        gt = normalize_letter(q.get("gt_answer"))
        if gt is None:
            continue
        q_type = str(q.get("question_type", "unknown"))
        s = scores.setdefault(q_type, {"total": 0, "answered": 0, "correct": 0})
        s["total"] += 1
        pred = normalize_letter(q.get(pred_field))
        if pred is not None:
            s["answered"] += 1
            if pred == gt:
                s["correct"] += 1
    return scores


def evaluate_benchmark_predictions(payload: dict[str, Any], pred_field: str) -> dict[str, Any]:
    by_condition: dict[str, dict[str, int]] = {}
    by_type:      dict[str, dict[str, int]] = {}
    by_type_cond: dict[str, dict[str, dict[str, int]]] = {}
    paired_drops: dict[str, dict[str, list[float]]] = {}

    def _add(d: dict[str, int], s: dict[str, int]) -> None:
        for k in ("total", "answered", "correct"):
            d[k] = d.get(k, 0) + s[k]

    for scene in payload.get("scenes") or []:
        if not isinstance(scene, dict):
            continue
        for target in scene.get("targets") or []:
            if not isinstance(target, dict):
                continue

            clean_block = target.get("clean")
            clean_scores: dict[str, dict[str, int]] = {}
            if isinstance(clean_block, dict):
                clean_scores = _score_block(clean_block, pred_field)
                for q_type, s in clean_scores.items():
                    _add(by_condition.setdefault("clean", {}), s)
                    _add(by_type.setdefault(q_type, {}), s)
                    _add(by_type_cond.setdefault(q_type, {}).setdefault("clean", {}), s)

            for occ in target.get("occlusions") or []:
                if not isinstance(occ, dict):
                    continue
                occ_type  = occ.get("occlusion_type", "unknown")
                occ_scores = _score_block({"questions": occ.get("questions")}, pred_field)
                for q_type, s in occ_scores.items():
                    _add(by_condition.setdefault(occ_type, {}), s)
                    _add(by_type.setdefault(q_type, {}), s)
                    _add(by_type_cond.setdefault(q_type, {}).setdefault(occ_type, {}), s)
                    cs = clean_scores.get(q_type)
                    if cs and cs.get("answered") and s.get("answered"):
                        clean_acc = cs["correct"] / cs["answered"]
                        occ_acc   = s["correct"]  / s["answered"]
                        paired_drops.setdefault(occ_type, {}).setdefault(q_type, []).append(
                            clean_acc - occ_acc
                        )

    paired_drop_summary: dict[str, Any] = {}
    for occ_type, qt_map in paired_drops.items():
        all_vals = [v for vals in qt_map.values() for v in vals]
        paired_drop_summary[occ_type] = {
            "avg_drop": sum(all_vals) / len(all_vals) if all_vals else None,
            "by_type":  {qt: sum(v) / len(v) for qt, v in qt_map.items() if v},
        }

    total    = sum(s.get("total",    0) for s in by_condition.values())
    answered = sum(s.get("answered", 0) for s in by_condition.values())
    correct  = sum(s.get("correct",  0) for s in by_condition.values())

    return {
        "field": pred_field,
        "total": total, "answered": answered, "correct": correct,
        "coverage":     (answered / total)    if total    else 0.0,
        "acc_all":      (correct  / total)    if total    else 0.0,
        "acc_answered": (correct  / answered) if answered else 0.0,
        "by_condition": by_condition,
        "by_type":      by_type,
        "by_type_cond": by_type_cond,
        "paired_drop":  paired_drop_summary,
    }


def _pct(s: dict[str, int]) -> str:
    a, c = s.get("answered", 0), s.get("correct", 0)
    return f"{c/a:.1%}" if a else "n/a"


def print_benchmark_report(report: dict[str, Any]) -> None:
    print(f"\nField: {report['field']} (occlusion benchmark)")
    print("-" * 60)
    print(f"total={report['total']}  answered={report['answered']}  correct={report['correct']}")
    print(f"coverage={report['coverage']:.1%}  acc(all)={report['acc_all']:.1%}  acc(answered)={report['acc_answered']:.1%}")

    print("\nby_condition:")
    for cond, s in sorted(report.get("by_condition", {}).items()):
        t, a, c = s.get("total",0), s.get("answered",0), s.get("correct",0)
        print(f"  {cond:25s}  total={t:4d}  answered={a:4d}  correct={c:4d}  acc={_pct(s)}")

    print("\nby_question_type:")
    for q_type, s in sorted(report.get("by_type", {}).items()):
        t, a, c = s.get("total",0), s.get("answered",0), s.get("correct",0)
        print(f"  {q_type:25s}  total={t:4d}  answered={a:4d}  correct={c:4d}  acc={_pct(s)}")

    by_tc = report.get("by_type_cond", {})
    if by_tc:
        all_conds = sorted({c for qt in by_tc.values() for c in qt})
        header = "  " + f"{'':25s}" + "".join(f"  {c:>18s}" for c in all_conds)
        print(f"\nby_question_type × by_condition:")
        print(header)
        for q_type in sorted(by_tc):
            row = f"  {q_type:25s}"
            for cond in all_conds:
                s = by_tc[q_type].get(cond, {})
                row += f"  {_pct(s):>18s}"
            print(row)

    paired = report.get("paired_drop", {})
    if paired:
        print("\npaired drop vs clean (clean_acc - occ_acc, per unit, averaged):")
        for occ_type in sorted(paired):
            d = paired[occ_type]
            avg = d.get("avg_drop")
            avg_s = f"{avg:.1%}" if avg is not None else "n/a"
            print(f"  clean → {occ_type:20s}  avg_drop = {avg_s}")
            for qt, dv in sorted(d.get("by_type", {}).items()):
                print(f"    {qt:23s}  {dv:.1%}")


# ── Utilities ─────────────────────────────────────────────────────────────────

def normalize_letter(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    s = value.strip().upper()
    if not s:
        return None
    for c in s:
        if c in VALID_LETTERS:
            return c
    return None


def list_images(multiview_dir: Path, max_images: int) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    files = [p for p in multiview_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)[:max_images]


def sample_video_frames(video_path: Path, n_frames: int) -> list[Path]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg first.")
    tmp_dir = Path(tempfile.mkdtemp(prefix="gemini_frames_"))
    out_pattern = tmp_dir / "frame_%03d.jpg"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(video_path), "-frames:v", str(n_frames), str(out_pattern)]
    subprocess.run(cmd, check=True)
    frames = sorted(tmp_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError("No frames extracted from video.")
    return frames[:n_frames]


def image_to_data(path: Path) -> tuple[bytes, str]:
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "image/png"
    return path.read_bytes(), mime


def iter_questions(payload: dict[str, Any]):
    records = payload.get("records", [])
    if isinstance(records, list) and records:
        for rec in records:
            if not isinstance(rec, dict):
                continue
            for q in rec.get("questions") or []:
                if isinstance(q, dict):
                    yield q
        return
    flat = payload.get("questions")
    if isinstance(flat, list):
        for q in flat:
            if isinstance(q, dict):
                yield q


# ── Gemini client & inference ─────────────────────────────────────────────────

def make_gemini_client(model: str):
    try:
        from google import genai          # type: ignore
        from google.genai import types    # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: google-genai. Install with `pip install google-genai`.") from e
    api_key = load_gemini_api_key()
    client  = genai.Client(api_key=api_key)
    return client, model, types


def build_prompt(question: str, answer_options: list[str], prompt_style: str = "default",
                 text_only: bool = False) -> str:
    options_text = "\n".join(answer_options)
    valid = ", ".join(VALID_LETTERS[:len(answer_options)])

    if prompt_style == "cot":
        cannot_letter = ""
        for opt in answer_options:
            if "cannot determine" in opt.lower():
                cannot_letter = opt.split(")")[0].strip()
                break
        cannot_ref = f"{cannot_letter})" if cannot_letter else "the 'Cannot determine' option"
        if text_only:
            return (
                "You are answering a multiple-choice question.\n\n"
                "Follow these steps before selecting your answer:\n\n"
                "Step 1: Based on the question context, can the target object's properties be inferred? Answer Yes or No.\n"
                "Step 2: Is the context reliable for answering the question? Answer Yes or No.\n"
                f"Step 3: If both answers are Yes, select the correct option.\n"
                f"        Otherwise, select {cannot_ref} Cannot determine.\n\n"
                f"Question:\n{question}\n\n"
                f"Options:\n{options_text}\n\n"
                f"Write your step-by-step reasoning, then on a new line write:\n"
                f"Answer: <letter from {valid}>"
            )
        return (
            "You are answering a visual multiple-choice question about the provided image(s).\n\n"
            "Follow these steps before selecting your answer:\n\n"
            "Step 1: Is the target object fully visible? Answer Yes or No.\n"
            "Step 2: Is the viewpoint reliable for answering the question? Answer Yes or No.\n"
            f"Step 3: If both answers are Yes, select the correct option.\n"
            f"        Otherwise, select {cannot_ref} Cannot determine.\n\n"
            f"Question:\n{question}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Write your step-by-step reasoning, then on a new line write:\n"
            f"Answer: <letter from {valid}>"
        )

    if text_only:
        return (
            "You are answering a multiple-choice question.\n"
            "Choose the best answer based on your knowledge.\n\n"
            "Rules:\n"
            "- You must choose exactly one option.\n"
            "- Reply with ONLY one letter.\n\n"
            f"Question:\n{question}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Reply with ONLY one letter from: {valid}."
        )
    return (
        "You are answering a visual multiple-choice question about the provided image(s).\n"
        "Look carefully at the image and choose the best answer based on visible evidence.\n\n"
        "Rules:\n"
        "- You must choose exactly one option.\n"
        "- Choose 'Cannot determine' if the image lacks sufficient visual evidence to decide reliably.\n"
        "- Reply with ONLY one letter.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Reply with ONLY one letter from: {valid}."
    )


def _get_response_text(resp) -> tuple[str, str]:
    text = getattr(resp, "text", None)
    if text is not None and str(text).strip():
        return str(text).strip(), "ok:resp.text"
    for c in (getattr(resp, "candidates", None) or []):
        parts_text = [
            str(getattr(p, "text", "")).strip()
            for p in (getattr(getattr(c, "content", None), "parts", None) or [])
            if getattr(p, "text", None)
        ]
        if parts_text:
            reason = getattr(c, "finish_reason", None)
            return "\n".join(parts_text), f"ok:candidate.parts finish={reason}"
    feedback = getattr(resp, "prompt_feedback", None)
    if feedback is not None:
        block = getattr(feedback, "block_reason", None)
        if block is not None:
            return "", f"empty:block_reason={block}"
    if getattr(resp, "candidates", None):
        reason = getattr(resp.candidates[0], "finish_reason", None)
        return "", f"empty:finish_reason={reason}"
    return "", "empty:no_text_no_candidates"


def _is_retryable_gemini_error(e: Exception) -> bool:
    code = getattr(e, "status_code", None)
    if code in (429, 500, 502, 503, 504):
        return True
    msg = str(e).lower()
    return (
        "503" in msg
        or "unavailable" in msg
        or "high demand" in msg
        or "overloaded" in msg
        or "rate limit" in msg
        or "429" in msg
        or "temporar" in msg
        or "deadline" in msg
        or "timeout" in msg
        or "connection reset" in msg
        or "connection aborted" in msg
    )


def _retry_sleep_s(attempt: int, base_s: float, max_s: float) -> float:
    sleep_s = min(max_s, base_s * (2 ** (attempt - 1)))
    return sleep_s * (0.75 + 0.5 * random.random())


def _gemini_generate_with_retry(
    client,
    model_name: str,
    contents: list[Any],
    config: Any,
    *,
    max_retries: int,
    retry_base_s: float,
    retry_max_s: float,
) -> Any:
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 2):  # +1 final try
        try:
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as e:
            last_err = e
            if attempt >= max_retries + 1 or not _is_retryable_gemini_error(e):
                raise
            sleep_s = _retry_sleep_s(attempt, retry_base_s, retry_max_s)
            print(
                f"[WARN] Gemini transient error (attempt {attempt}/{max_retries}) "
                f"-> retry in {sleep_s:.1f}s: {e}"
            )
            time.sleep(sleep_s)
    raise RuntimeError(f"Gemini generate_content failed after retries: {last_err}")


def ask_one_question(
    client,
    model_name: str,
    types_mod,
    question: str,
    answer_options: list[str],
    image_paths: list[Path],
    prompt_style: str = "default",
) -> str:
    missing = [str(p) for p in image_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Image paths do not exist: {missing[:5]}")

    text_only = not image_paths
    prompt_text = build_prompt(question, answer_options, prompt_style, text_only=text_only)
    image_parts: list[Any] = []
    for p in image_paths:
        data, mime = image_to_data(p)
        image_parts.append(types_mod.Part.from_bytes(data=data, mime_type=mime))

    contents = [*image_parts, types_mod.Part.from_text(text=prompt_text)]
    max_tokens = 1024 if prompt_style == "cot" else 512
    cfg_kwargs: dict[str, Any] = {"temperature": 0.0, "max_output_tokens": max_tokens}
    try:
        thinking_cfg = getattr(types_mod, "ThinkingConfig", None)
        if thinking_cfg is not None:
            cfg_kwargs["thinking_config"] = thinking_cfg(thinking_budget=0)
    except Exception:
        pass

    try:
        resp = _gemini_generate_with_retry(
            client,
            model_name,
            contents,
            types_mod.GenerateContentConfig(**cfg_kwargs),
            max_retries=int(getattr(ask_one_question, "_max_retries", 50)),
            retry_base_s=float(getattr(ask_one_question, "_retry_base_s", 2.0)),
            retry_max_s=float(getattr(ask_one_question, "_retry_max_s", 60.0)),
        )
    except Exception as e:
        msg = str(e)
        if "Budget 0 is invalid" in msg or "only works in thinking mode" in msg:
            cfg_kwargs.pop("thinking_config", None)
            resp = _gemini_generate_with_retry(
                client,
                model_name,
                contents,
                types_mod.GenerateContentConfig(**cfg_kwargs),
                max_retries=int(getattr(ask_one_question, "_max_retries", 50)),
                retry_base_s=float(getattr(ask_one_question, "_retry_base_s", 2.0)),
                retry_max_s=float(getattr(ask_one_question, "_retry_max_s", 60.0)),
            )
        else:
            raise

    text, _diag = _get_response_text(resp)

    if prompt_style == "cot":
        import re
        matches = re.findall(r"Answer\s*:\s*([A-Ja-j])", text)
        if matches:
            return matches[-1].upper()
        matches2 = re.findall(r"\b([A-Ja-j])\b", text)
        if matches2:
            return matches2[-1].upper()
        return ""

    letter = normalize_letter(text)
    if letter:
        return letter

    # Retry with stricter prompt
    strict = (
        f"Question:\n{question}\n\n"
        f"Options:\n{chr(10).join(answer_options)}\n\n"
        "Return exactly one uppercase letter only."
    )
    resp2 = _gemini_generate_with_retry(
        client,
        model_name,
        [*image_parts, types_mod.Part.from_text(text=strict)],
        types_mod.GenerateContentConfig(temperature=0.0, max_output_tokens=512),
        max_retries=int(getattr(ask_one_question, "_max_retries", 50)),
        retry_base_s=float(getattr(ask_one_question, "_retry_base_s", 2.0)),
        retry_max_s=float(getattr(ask_one_question, "_retry_max_s", 60.0)),
    )
    text2, _diag2 = _get_response_text(resp2)
    return normalize_letter(text2) or ""


# ── run_modality ──────────────────────────────────────────────────────────────

def run_modality(
    payload: dict[str, Any],
    client,
    model_name: str,
    types_mod,
    image_paths: list[Path],
    pred_field: str,
    overwrite: bool,
    limit: int,
    progress_bar: Any | None = None,
    rerun_question_types: set[str] | None = None,
    preds_jsonl_path: Path | None = None,
    prompt_style: str = "default",
) -> int:
    todo: list[tuple[dict[str, Any], str, list[str]]] = []
    for q in iter_questions(payload):
        q_type = str(q.get("question_type", ""))
        must_rerun = bool(rerun_question_types and q_type in rerun_question_types)
        if not overwrite and normalize_letter(q.get(pred_field)) and not must_rerun:
            continue
        question = str(q.get("question", "")).strip()
        options_any = q.get("answer_options")
        if not question or not isinstance(options_any, list) or not options_any:
            continue
        todo.append((q, question, [str(x) for x in options_any]))

    if limit > 0:
        todo = todo[:limit]
    total = len(todo)
    if total == 0:
        print(f"[{pred_field}] Nothing to do.")
        return 0

    use_tqdm = False
    bar = None
    if progress_bar is not None:
        bar, use_tqdm = progress_bar, True
    else:
        try:
            from tqdm.auto import tqdm  # type: ignore
            bar, use_tqdm = tqdm(total=total, desc=pred_field, unit="q"), True
        except Exception:
            pass

    written = 0
    ans_count   = 0
    unans_count = 0
    t0 = time.time()
    last_print = t0

    def _is_unans(pred: str, options: list[str]) -> bool:
        for opt in options:
            letter = opt.split(")")[0].strip().upper()
            if letter == pred.upper() and "cannot determine" in opt.lower():
                return True
        return False

    for idx, (q, question, options) in enumerate(todo, start=1):
        pred = ask_one_question(client, model_name, types_mod, question, options, image_paths, prompt_style=prompt_style)
        q[pred_field] = pred
        written += 1
        if pred and _is_unans(pred, options):
            unans_count += 1
        elif pred:
            ans_count += 1
        if preds_jsonl_path is not None:
            qid = _question_stable_id(q)
            if not qid:
                raise RuntimeError(
                    f"Missing question_id; cannot append JSONL for {pred_field!r}. "
                    "Ensure benchmark questions include stable question_id fields."
                )
            append_pred_jsonl(preds_jsonl_path, {"question_id": qid, "pred_field": pred_field, "pred": pred})

        if use_tqdm and bar is not None:
            bar.update(1)
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(ans=ans_count, unans=unans_count)
        else:
            now = time.time()
            if idx == 1 or idx == total or (now - last_print) >= 2.0:
                elapsed = max(1e-6, now - t0)
                rate = idx / elapsed
                eta  = (total - idx) / max(1e-6, rate)
                print(
                    f"[{pred_field}] {idx}/{total} ({idx/total*100:.1f}%) "
                    f"ans={ans_count} unans={unans_count} "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s"
                )
                last_print = now

    if use_tqdm and bar is not None and progress_bar is None and hasattr(bar, "close"):
        bar.close()

    return written


def _count_block_todo_questions(
    block: dict[str, Any],
    pred_field: str,
    overwrite: bool,
    rerun_question_types: set[str] | None = None,
) -> int:
    questions = block.get("questions")
    if not isinstance(questions, list) or not questions:
        return 0
    n = 0
    for q in questions:
        if not isinstance(q, dict):
            continue
        q_type = str(q.get("question_type", ""))
        must_rerun = bool(rerun_question_types and q_type in rerun_question_types)
        if not overwrite and normalize_letter(q.get(pred_field)) and not must_rerun:
            continue
        if not str(q.get("question", "")).strip():
            continue
        options_any = q.get("answer_options")
        if not isinstance(options_any, list) or not options_any:
            continue
        n += 1
    return n


# ── process_occlusion_benchmark ───────────────────────────────────────────────

def process_occlusion_benchmark(
    benchmark_path: Path,
    payload: dict[str, Any],
    args: Any,
    client: Any,
    model_name: str,
    types_mod: Any,
) -> None:
    anchor = benchmark_path_anchor(payload, benchmark_path)
    print(f"Occlusion benchmark media anchor: {anchor}")

    conditions: list[str] = []
    if args.benchmark_condition in ("both", "clean"):
        conditions.append("clean")
    if args.benchmark_condition in ("both", "partial_occlusion"):
        conditions.append("partial_occlusion")
    if args.benchmark_condition in ("both", "full_occlusion"):
        conditions.append("full_occlusion")

    rerun_types = _parse_question_types_csv(getattr(args, "rerun_question_types", "") or "")

    def _count_benchmark_total(modality_inner: str, pred_field: str, dataset: dict[str, Any] | None = None) -> int:
        src = dataset if dataset is not None else payload
        total = 0
        for scene in src.get("scenes") or []:
            if not isinstance(scene, dict):
                continue
            for target in scene.get("targets") or []:
                if not isinstance(target, dict):
                    continue
                for cond, uid, block in _iter_target_blocks(target, conditions):
                    if modality_inner == "text":
                        pass  # text-only: no media required
                    elif modality_inner in ("multiview", "singleview"):
                        image_paths = collect_multiview_paths(block, anchor, args.max_images)
                        if modality_inner == "singleview":
                            ora = oracle_image_path(block, anchor)
                            image_paths = [ora] if ora and ora.is_file() else []
                        if not image_paths:
                            raise RuntimeError(
                                f"Missing images for {scene.get('scene_id')} | {uid} | {cond} ({modality_inner}). "
                                f"Check benchmark media anchor: {anchor}"
                            )
                    else:
                        vid = walkthrough_path_for_block(block, anchor)
                        if vid is None or not vid.is_file():
                            raise RuntimeError(
                                f"Missing walkthrough video for {scene.get('scene_id')} | {uid} | {cond}."
                            )
                    total += _count_block_todo_questions(block, pred_field, args.overwrite, rerun_types)
        return total

    bar: Any | None = None

    def fill_payload(
        work: dict[str, Any],
        modality_inner: str,
        pred_field: str = PRED_FIELD,
        limit_left: list[int | None] | None = None,
        preds_jsonl_path: Path | None = None,
    ) -> None:
        video_cache: dict[str, list[Path]] = {}
        for scene in work.get("scenes") or []:
            if not isinstance(scene, dict):
                continue
            for target in scene.get("targets") or []:
                if not isinstance(target, dict):
                    continue
                for cond, uid, block in _iter_target_blocks(target, conditions):
                    questions = block.get("questions")
                    if not isinstance(questions, list) or not questions:
                        continue
                    if not args.overwrite and not rerun_types and all(
                        normalize_letter(q.get(pred_field)) for q in questions if isinstance(q, dict)
                    ):
                        continue

                    if modality_inner == "text":
                        image_paths = []  # text-only: no images
                    elif modality_inner in ("multiview", "singleview"):
                        image_paths = collect_multiview_paths(block, anchor, args.max_images)
                        if modality_inner == "singleview":
                            ora = oracle_image_path(block, anchor)
                            image_paths = [ora] if ora and ora.is_file() else []
                        if not image_paths:
                            raise RuntimeError(
                                f"Missing images for {scene.get('scene_id')} | {uid} | {cond} ({modality_inner})."
                            )
                    else:
                        vid = walkthrough_path_for_block(block, anchor)
                        if vid is None or not vid.is_file():
                            raise RuntimeError(
                                f"Missing walkthrough.mp4 for {scene.get('scene_id')} | {uid} | {cond}."
                            )
                        vkey = str(vid)
                        if vkey not in video_cache:
                            video_cache[vkey] = sample_video_frames(vid, args.video_frames)
                        image_paths = video_cache[vkey]
                        if not image_paths:
                            raise RuntimeError(
                                f"No video frames extracted for {scene.get('scene_id')} | {uid} | {cond}."
                            )

                    rec_id = f"{scene.get('scene_id')}|{uid}|{cond}"
                    mini: dict[str, Any] = {"records": [{"target_id": rec_id, "questions": questions}]}
                    lim = 0
                    if limit_left is not None:
                        if limit_left[0] is not None and limit_left[0] <= 0:
                            if bar is not None and hasattr(bar, "close"):
                                bar.close()
                            return
                        lim = 0 if limit_left[0] is None else limit_left[0]

                    n = run_modality(
                        mini, client, model_name, types_mod, image_paths,
                        pred_field, args.overwrite, lim,
                        progress_bar=bar,
                        rerun_question_types=rerun_types,
                        preds_jsonl_path=preds_jsonl_path,
                        prompt_style=getattr(args, "prompt_style", "default"),
                    )
                    if limit_left is not None and limit_left[0] is not None:
                        limit_left[0] = max(0, limit_left[0] - n)
                        if limit_left[0] <= 0:
                            if bar is not None and hasattr(bar, "close"):
                                bar.close()
                            return

    # ── Single modality ───────────────────────────────────────────────────────
    out_path = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else default_benchmark_output_path(benchmark_path, args.model, args.modality)
    )
    preds_jsonl = resolve_benchmark_preds_jsonl_path(out_path, getattr(args, "predictions_jsonl", "") or "")

    if args.overwrite and preds_jsonl.is_file():
        preds_jsonl.unlink()

    work = json.loads(json.dumps(payload))

    if not args.overwrite and not preds_jsonl.is_file() and out_path.is_file():
        n_mig = export_benchmark_preds_to_jsonl(out_path, preds_jsonl, [PRED_FIELD], conditions)
        print(f"[INFO] Migrated {n_mig} predictions from existing eval JSON -> {preds_jsonl}")

    preds_map = load_preds_jsonl_map(preds_jsonl)
    merge_preds_map_into_benchmark_work(work, preds_map, conditions, [PRED_FIELD])

    if rerun_types:
        cleared, cleared_ids = _clear_benchmark_predictions_for_types(work, [PRED_FIELD], rerun_types, conditions)
        print(f"[INFO] Cleared predictions for rerun types {sorted(rerun_types)}: {cleared} field(s)")
        rm = filter_jsonl_drop_question_ids(preds_jsonl, cleared_ids)
        print(f"[INFO] Removed {rm} JSONL lines for those question_ids")

    already = _count_existing_predictions(work, PRED_FIELD)
    print(
        f"[INFO] Benchmark; preds sidecar: {preds_jsonl} "
        f"(already={already}) full eval -> {out_path}"
    )

    try:
        from tqdm.auto import tqdm
        computed_rem = _count_benchmark_total(args.modality, PRED_FIELD, work)
        total_for_bar = (args.limit if args.limit > 0 else computed_rem) or 0
        if total_for_bar > 0:
            bar = tqdm(total=total_for_bar, desc="benchmark", unit="q")
            if args.limit > 0:
                try:
                    bar.update(min(already, args.limit))
                except Exception:
                    pass
    except Exception:
        bar = None

    if args.limit > 0:
        remaining   = max(0, args.limit - already)
        limit_state: list[int | None] = [remaining]
    else:
        limit_state = [None]

    if limit_state[0] == 0:
        print(f"Skip benchmark (limit already satisfied): {out_path}")
        if bar is not None and hasattr(bar, "close"):
            bar.close()
        return

    label = {
        "singleview":  "singleview (oracle frame image per block)",
        "multiview":   "multiview",
        "walkthrough": "walkthrough",
        "text":        "text-only (no images sent to model)",
    }.get(args.modality, args.modality)
    print(f"Occlusion benchmark | {label}")

    fill_payload(work, args.modality, pred_field=PRED_FIELD, limit_left=limit_state, preds_jsonl_path=preds_jsonl)

    save_json(out_path, work)
    print(f"\nSaved: {out_path}")
    print_benchmark_report(evaluate_benchmark_predictions(work, PRED_FIELD))
    if bar is not None and hasattr(bar, "close"):
        bar.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini vision MC eval for the occlusion benchmark.")
    parser.add_argument(
        "--questions-json",
        default="/Users/zhangyue/Desktop/Holodeck/rendered_scene/occlusion_benchmark.json",
    )
    parser.add_argument("--model",       default="gemini-2.5-flash")
    parser.add_argument("--max-images",  type=int, default=8)
    parser.add_argument("--video-frames", type=int, default=8)
    parser.add_argument(
        "--modality",
        default="singleview",
        choices=["multiview", "singleview", "walkthrough", "text"],
        help="Choose one modality. 'text' sends no images (text-only baseline).",
    )
    parser.add_argument("--output-json",       default="")
    parser.add_argument("--predictions-jsonl", dest="predictions_jsonl", default="")
    parser.add_argument("--overwrite",         action="store_true")
    parser.add_argument("--limit", type=int,   default=0)
    parser.add_argument(
        "--rerun-question-types",
        dest="rerun_question_types",
        default="",
        help="Comma-separated question_type values to re-evaluate. Example: relative_position",
    )
    parser.add_argument(
        "--benchmark-condition",
        default="both",
        choices=["both", "clean", "partial_occlusion", "full_occlusion"],
    )
    parser.add_argument(
        "--prompt-style",
        dest="prompt_style",
        default="default",
        choices=["default", "cot"],
        help="Prompt strategy: 'default' = direct MC; 'cot' = chain-of-thought reasoning first.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=50,
        help="Max retry attempts for transient Gemini errors (503/429/UNAVAILABLE/high demand). Default: 50.",
    )
    parser.add_argument(
        "--retry-base-s",
        type=float,
        default=2.0,
        help="Base backoff seconds for retries (exponential). Default: 2.0.",
    )
    parser.add_argument(
        "--retry-max-s",
        type=float,
        default=60.0,
        help="Max sleep seconds between retries. Default: 60.0.",
    )
    args = parser.parse_args()

    # Configure retry behavior for ask_one_question()
    ask_one_question._max_retries = int(args.max_retries)       # type: ignore[attr-defined]
    ask_one_question._retry_base_s = float(args.retry_base_s)   # type: ignore[attr-defined]
    ask_one_question._retry_max_s = float(args.retry_max_s)     # type: ignore[attr-defined]

    if args.prompt_style != "default" and not args.output_json:
        args._model_for_output = f"{args.model}_{args.prompt_style}"
    else:
        args._model_for_output = args.model

    client, model_name, types_mod = make_gemini_client(args.model)
    print(f"Backend: gemini, model: {model_name}")

    benchmark_path = Path(args.questions_json).expanduser().resolve()
    if not benchmark_path.is_file():
        raise FileNotFoundError(f"Benchmark JSON not found: {benchmark_path}")

    payload = load_json(benchmark_path)
    if not is_occlusion_benchmark(payload):
        raise ValueError("Input JSON does not look like an occlusion benchmark.")

    process_occlusion_benchmark(benchmark_path, payload, args, client, model_name, types_mod)


if __name__ == "__main__":
    main()
