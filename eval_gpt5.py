from __future__ import annotations

import argparse
import base64
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
MIN_AZURE_RESPONSES_API_VERSION = "2025-03-01-preview"
PRED_FIELD = "pred_answer"


def _read_key_from_file(path: Path) -> str:
    """Read a key from a file, allowing KEY=... format (same as other scripts)."""
    s = path.read_text(encoding="utf-8").strip()
    if "=" in s and s.split("=", 1)[0].strip().isupper():
        s = s.split("=", 1)[1].strip().strip("\"'")
    return s


def _load_azure_api_key_from_file() -> str | None:
    """
    Fallback loader for Azure API key, mirroring the behavior used in other scripts:
    look for a `.azure_key` file next to this script or at project root.
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    for d in (script_dir, project_root):
        p = d / ".azure_key"
        if p.is_file():
            try:
                key = _read_key_from_file(p).strip()
                if key:
                    return key
            except Exception:
                continue
    return None


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


def _question_stable_id(q: dict[str, Any]) -> str | None:
    """Stable key for merging predictions across benchmark edits (prefer question_id)."""
    rid = q.get("question_id") if q.get("question_id") is not None else q.get("id")
    if rid is None:
        return None
    s = str(rid).strip()
    return s or None


def default_benchmark_preds_jsonl_path(out_json: Path) -> Path:
    """Sidecar JSONL next to the benchmark eval JSON: stem_preds.jsonl"""
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
    """Map (question_id, pred_field) -> pred letter (last line wins)."""
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
            pf = str(obj.get("pred_field", "")).strip()
            pred = str(obj.get("pred", "")).strip()
            if qid and pf:
                m[(qid, pf)] = pred
    return m


def merge_preds_map_into_benchmark_work(
    work: dict[str, Any],
    preds: dict[tuple[str, str], str],
    conditions: tuple[str, ...] | list[str],
    pred_fields: list[str],
) -> int:
    """Apply preds map onto benchmark work in-place. Returns number of fields set."""
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
    conditions: tuple[str, ...] | list[str],
) -> int:
    """One-time migration: copy preds from an existing eval JSON into JSONL."""
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
                            lines.append(
                                json.dumps(
                                    {"question_id": qid, "pred_field": pf, "pred": letter},
                                    ensure_ascii=False,
                                )
                            )
    if lines:
        dst_jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def filter_jsonl_drop_question_ids(path: Path, drop_ids: set[str]) -> int:
    """Remove JSONL lines whose question_id is in drop_ids. Returns lines removed."""
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
    """Parse comma-separated question_type values into a non-empty set."""
    parts = [p.strip() for p in (s or "").split(",")]
    return {p for p in parts if p}


def _clear_benchmark_predictions_for_types(
    work: dict[str, Any],
    pred_fields: list[str],
    types: set[str],
    conditions: tuple[str, ...] | list[str],
) -> tuple[int, set[str]]:
    """
    Remove pred_* fields for questions whose question_type is in `types`.
    Returns (how many (question, pred_field) pairs were cleared, question_ids touched).
    """
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
    # Benchmark payload uses scenes→targets→blocks; non-benchmark uses records/questions.
    if is_occlusion_benchmark(payload):
        n = 0
        for scene in payload.get("scenes") or []:
            if not isinstance(scene, dict):
                continue
            for target in scene.get("targets") or []:
                if not isinstance(target, dict):
                    continue
                # Count across all condition blocks present in the file.
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


def _safe_name(s: str) -> str:
    """Filename-safe short token."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in (s or "").strip())


def default_output_path(input_questions_json: Path, model: str, modality: str) -> Path:
    """
    Default output name: target_questions_{model}_{modality}.json
    Saved next to the input questions json.
    """
    model_tok = _safe_name(model) or "model"
    modality_tok = _safe_name(modality) or "modality"
    return input_questions_json.with_name(f"target_questions_{model_tok}_{modality_tok}.json")


def default_benchmark_output_path(benchmark_json: Path, model: str, modality: str) -> Path:
    model_tok = _safe_name(model) or "model"
    modality_tok = _safe_name(modality) or "modality"
    stem = benchmark_json.stem if benchmark_json.suffix else benchmark_json.name
    return benchmark_json.with_name(f"{stem}_eval_{model_tok}_{modality_tok}.json")


def default_benchmark_merged_output_path(benchmark_json: Path, model: str) -> Path:
    """Single file when --modality both on occlusion_benchmark (multiview + walkthrough preds)."""
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
    p = of.get("image_path")
    r = resolve_benchmark_media_path(str(p) if p else None, anchor)
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
    """Yield (cond, unit_id, block) for each condition block in a target.

    New structure:
      target.clean                     → cond="clean"
      target.occlusions[].occlusion_type / scene_dir / image_retrieval / questions
    """
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


def _score_block(block: dict[str, Any], pred_field: str) -> dict[str, dict[str, int]]:
    """Return {question_type: {total, answered, correct}} for one condition block."""
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


def _acc(s: dict[str, int]) -> float | None:
    return (s["correct"] / s["total"]) if s["total"] else None


def evaluate_benchmark_predictions(payload: dict[str, Any], pred_field: str) -> dict[str, Any]:
    """
    Paired evaluation: for each (target, occlusion), score clean and occlusion
    conditions independently. Reports:
      - accuracy by condition
      - accuracy by question type
      - accuracy by question_type × condition
      - paired drop per unit (controls for scene difficulty)
    """
    by_condition: dict[str, dict[str, int]] = {}
    by_type:      dict[str, dict[str, int]] = {}
    # by_type_cond[q_type][cond] = {total, answered, correct}
    by_type_cond: dict[str, dict[str, dict[str, int]]] = {}
    # paired_drops[occ_type][q_type] = [drop_per_unit, ...]  (clean_acc - occ_acc per unit)
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
                occ_type = occ.get("occlusion_type", "unknown")
                occ_scores = _score_block({"questions": occ.get("questions")}, pred_field)

                for q_type, s in occ_scores.items():
                    _add(by_condition.setdefault(occ_type, {}), s)
                    _add(by_type.setdefault(q_type, {}), s)
                    _add(by_type_cond.setdefault(q_type, {}).setdefault(occ_type, {}), s)

                    # Paired drop: clean_acc - occ_acc for this unit
                    cs = clean_scores.get(q_type)
                    if cs and cs.get("answered") and s.get("answered"):
                        clean_acc = cs["correct"] / cs["answered"]
                        occ_acc   = s["correct"]  / s["answered"]
                        paired_drops.setdefault(occ_type, {}).setdefault(q_type, []).append(
                            clean_acc - occ_acc
                        )

    # Summarise paired drops
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
        "total": total,
        "answered": answered,
        "correct": correct,
        "coverage":      (answered / total)    if total    else 0.0,
        "acc_all":       (correct  / total)    if total    else 0.0,
        "acc_answered":  (correct  / answered) if answered else 0.0,
        "by_condition":  by_condition,
        "by_type":       by_type,
        "by_type_cond":  by_type_cond,
        "paired_drop":   paired_drop_summary,
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

    # by_question_type × by_condition
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

    # Paired drop
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
    tmp_dir = Path(tempfile.mkdtemp(prefix="gpt5_video_frames_"))
    out_pattern = tmp_dir / "frame_%03d.jpg"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-frames:v",
        str(n_frames),
        str(out_pattern),
    ]
    subprocess.run(cmd, check=True)
    frames = sorted(tmp_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError("No frames extracted from video.")
    return frames[:n_frames]


def image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "image/png"
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_prompt(question: str, answer_options: list[str], model_name: str = "",
                 prompt_style: str = "default", text_only: bool = False) -> str:
    options_text = "\n".join(answer_options)
    valid = ", ".join(VALID_LETTERS[: len(answer_options)])
    visual_prefix = "" if text_only else "visual "
    image_clause = "" if text_only else "Look carefully at the image and choose the best answer based on visible evidence.\n"

    if prompt_style == "cot":
        # Find which letter corresponds to "Cannot determine"
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
                "Step 2: Is the viewpoint or context reliable for answering the question? Answer Yes or No.\n"
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

    # default prompt
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


def make_client(provider: str, model: str):
    try:
        from openai import OpenAI, AzureOpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: openai. Install with `pip install openai`.") from e

    provider = provider.lower().strip()
    if provider not in {"auto", "openai", "azure"}:
        raise ValueError("provider must be one of: auto, openai, azure")

    if provider in {"auto", "azure"}:
        # Hard-coded Azure config, same style as azure_test.py / answer_with_model_*.py
        endpoint = "https://murge-foundry-yue.cognitiveservices.azure.com/"
        api_key = (
            os.getenv("AZURE_OPENAI_API_KEY")
            or _load_azure_api_key_from_file()
        )
        # Responses API requires >= 2025-03-01-preview on Azure.
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", MIN_AZURE_RESPONSES_API_VERSION)
        if api_version < MIN_AZURE_RESPONSES_API_VERSION:
            print(
                f"[Warn] AZURE_OPENAI_API_VERSION={api_version} is too old for Responses API; "
                f"using {MIN_AZURE_RESPONSES_API_VERSION}."
            )
            api_version = MIN_AZURE_RESPONSES_API_VERSION
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or model
        client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
        return client, deployment, "azure"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        key_file = Path(__file__).resolve().parent / ".openai_key"
        if key_file.is_file():
            api_key = key_file.read_text(encoding="utf-8").strip()
    if api_key:
        client = OpenAI(api_key=api_key)
        return client, model, "openai"
    raise RuntimeError("No API key found. Set OPENAI_API_KEY or Azure OpenAI env vars.")


def ask_one_question(client, model_name: str, question: str, answer_options: list[str],
                     image_paths: list[Path], prompt_style: str = "default") -> str:
    missing = [str(p) for p in image_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Some image paths do not exist: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    text_only = not image_paths
    content: list[dict[str, Any]] = [{"type": "input_text", "text": build_prompt(question, answer_options, model_name, prompt_style, text_only=text_only)}]
    for p in image_paths:
        content.append({"type": "input_image", "image_url": image_to_data_url(p)})

    resp = client.responses.create(
        model=model_name,
        input=[{"role": "user", "content": content}],
        #temperature=0,
    )
    text = (getattr(resp, "output_text", "") or "").strip()

    if prompt_style == "cot":
        import re
        # Parse "Answer: X" (last occurrence wins)
        matches = re.findall(r"Answer\s*:\s*([A-Ja-j])", text)
        if matches:
            return matches[-1].upper()
        # Fallback: last standalone letter in text
        matches2 = re.findall(r"\b([A-Ja-j])\b", text)
        if matches2:
            return matches2[-1].upper()
        return ""

    letter = normalize_letter(text)
    return letter or ""


def iter_questions(payload: dict[str, Any]):
    # Holodeck target_questions: { "records": [ { "questions": [...] } ] }
    records = payload.get("records", [])
    if isinstance(records, list) and records:
        for rec in records:
            if not isinstance(rec, dict):
                continue
            questions = rec.get("questions", [])
            if not isinstance(questions, list):
                continue
            for q in questions:
                if isinstance(q, dict):
                    yield q
        return
    # question_gen occlusion JSON: { "meta": ..., "questions": [...] }
    flat = payload.get("questions")
    if isinstance(flat, list):
        for q in flat:
            if isinstance(q, dict):
                yield q


def evaluate(payload: dict[str, Any], pred_field: str) -> dict[str, Any]:
    total = 0
    answered = 0
    correct = 0
    by_type: dict[str, dict[str, int]] = {}

    for q in iter_questions(payload):
        gt = normalize_letter(q.get("gt_answer"))
        if gt is None:
            continue
        total += 1
        q_type = str(q.get("question_type", "unknown"))
        stat = by_type.setdefault(q_type, {"total": 0, "answered": 0, "correct": 0})
        stat["total"] += 1
        pred = normalize_letter(q.get(pred_field))
        if pred is None:
            continue
        answered += 1
        stat["answered"] += 1
        if pred == gt:
            correct += 1
            stat["correct"] += 1

    return {
        "field": pred_field,
        "total": total,
        "answered": answered,
        "correct": correct,
        "coverage": (answered / total) if total else 0.0,
        "acc_all": (correct / total) if total else 0.0,
        "acc_answered": (correct / answered) if answered else 0.0,
        "by_type": by_type,
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"\nField: {report['field']}")
    print("-" * 60)
    print(f"total={report['total']} answered={report['answered']} correct={report['correct']}")
    print(f"coverage={report['coverage']:.1%}")
    print(f"accuracy(all)={report['acc_all']:.1%}")
    print(f"accuracy(answered)={report['acc_answered']:.1%}")
    print("by_type:")
    for q_type, s in sorted(report["by_type"].items()):
        t, a, c = s["total"], s["answered"], s["correct"]
        acc_all = c / t if t else 0.0
        print(f"  - {q_type}: total={t}, answered={a}, correct={c}, acc_all={acc_all:.1%}")


def run_modality(
    payload: dict[str, Any],
    client,
    model_name: str,
    image_paths: list[Path],
    pred_field: str,
    overwrite: bool,
    limit: int,
    progress_bar: Any | None = None,
    rerun_question_types: set[str] | None = None,
    preds_jsonl_path: Path | None = None,
    prompt_style: str = "default",
) -> int:
    """Run model on all questions in payload; returns number of new predictions written."""
    # Precompute eligible questions so we can show a tqdm-like progress: x/y (+ ETA).
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
        options = [str(x) for x in options_any]
        todo.append((q, question, options))

    if limit > 0:
        todo = todo[:limit]

    total = len(todo)
    if total == 0:
        print(f"[{pred_field}] Nothing to do (all answered or invalid questions).")
        return 0

    # Try tqdm if installed; otherwise fall back to periodic progress lines.
    # If a caller provides a shared tqdm bar (e.g., occlusion benchmark), we will use it
    # and avoid creating a nested per-block progress bar.
    use_tqdm = False
    bar = None
    if progress_bar is not None:
        bar = progress_bar
        use_tqdm = True
    else:
        try:
            from tqdm.auto import tqdm  # type: ignore

            bar = tqdm(total=total, desc=pred_field, unit="q")
            use_tqdm = True
        except Exception:
            pass

    written = 0
    ans_count   = 0
    unans_count = 0
    t0 = time.time()
    last_print = t0

    def _is_unans(pred: str, options: list[str]) -> bool:
        """Return True if pred letter maps to a 'Cannot determine' option."""
        for opt in options:
            letter = opt.split(")")[0].strip().upper()
            if letter == pred.upper() and "cannot determine" in opt.lower():
                return True
        return False

    for idx, (q, question, options) in enumerate(todo, start=1):
        pred = ask_one_question(client, model_name, question, options, image_paths, prompt_style=prompt_style)
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
                    f"Missing question_id on a question; cannot append JSONL row for {pred_field!r}. "
                    "Ensure benchmark questions include stable question_id fields."
                )
            append_pred_jsonl(
                preds_jsonl_path,
                {"question_id": qid, "pred_field": pred_field, "pred": pred},
            )

        if use_tqdm and bar is not None:
            bar.update(1)
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(ans=ans_count, unans=unans_count)
        else:
            now = time.time()
            if idx == 1 or idx == total or (now - last_print) >= 2.0:
                elapsed = max(1e-6, now - t0)
                rate = idx / elapsed
                eta = (total - idx) / max(1e-6, rate)
                pct = (idx / total) * 100.0
                print(
                    f"[{pred_field}] {idx}/{total} ({pct:.1f}%) "
                    f"ans={ans_count} unans={unans_count} "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s"
                )
                last_print = now

    if use_tqdm and bar is not None:
        if progress_bar is None and hasattr(bar, "close"):
            bar.close()

    return written


def _count_block_todo_questions(
    block: dict[str, Any],
    pred_field: str,
    overwrite: bool,
    rerun_question_types: set[str] | None = None,
) -> int:
    """Count questions in one block that would be evaluated by run_modality()."""
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
        question = str(q.get("question", "")).strip()
        options_any = q.get("answer_options")
        if not question or not isinstance(options_any, list) or not options_any:
            continue
        n += 1
    return n


def process_occlusion_benchmark(
    benchmark_path: Path,
    payload: dict[str, Any],
    args: Any,
    client: Any,
    model_name: str,
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

    def _count_benchmark_total(
        modality_inner: str, pred_field: str, dataset: dict[str, Any] | None = None
    ) -> int:
        """Count runnable questions for a modality (same media checks as fill_payload)."""
        src = dataset if dataset is not None else payload
        total = 0
        for scene in src.get("scenes") or []:
            if not isinstance(scene, dict):
                continue
            for target in scene.get("targets") or []:
                if not isinstance(target, dict):
                    continue
                for cond, uid, block in _iter_target_blocks(target, conditions):
                    # Mirror fill_payload's media existence checks so the total matches reality.
                    if modality_inner == "text":
                        pass  # text-only: no media required
                    elif modality_inner in ("multiview", "singleview"):
                        image_paths = collect_multiview_paths(block, anchor, args.max_images)
                        if modality_inner == "singleview":
                            ora = oracle_image_path(block, anchor)
                            image_paths = [ora] if ora and ora.is_file() else []
                        if not image_paths:
                            raise RuntimeError(
                                f"Missing images for {scene.get('scene_id')} | {uid} | {cond} "
                                f"({modality_inner}). Check benchmark media anchor: {anchor}"
                            )
                    else:
                        vid = walkthrough_path_for_block(block, anchor)
                        if vid is None or not vid.is_file():
                            raise RuntimeError(
                                f"Missing walkthrough video for {scene.get('scene_id')} | {uid} | {cond}. "
                                f"Expected: {vid}"
                            )

                    total += _count_block_todo_questions(
                        block, pred_field, args.overwrite, rerun_question_types=rerun_types
                    )
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
                                f"Missing images for {scene.get('scene_id')} | {uid} | {cond} "
                                f"({modality_inner}). This would run text-only, so we abort. "
                                f"Check benchmark media anchor: {anchor}"
                            )
                    else:
                        vid = walkthrough_path_for_block(block, anchor)
                        if vid is None or not vid.is_file():
                            raise RuntimeError(
                                f"Missing walkthrough.mp4 for {scene.get('scene_id')} | {uid} | {cond}. "
                                f"Check benchmark media anchor: {anchor}"
                            )
                        vkey = str(vid)
                        if vkey not in video_cache:
                            video_cache[vkey] = sample_video_frames(vid, args.video_frames)
                        image_paths = video_cache[vkey]
                        if not image_paths:
                            raise RuntimeError(
                                f"No video frames extracted for {scene.get('scene_id')} | {uid} | {cond}. "
                                f"Video: {vid}"
                            )

                    rec_id = f"{scene.get('scene_id')}|{uid}|{cond}"
                    mini: dict[str, Any] = {"records": [{"target_id": rec_id, "questions": questions}]}
                    if limit_left is None:
                        lim = 0
                    else:
                        if limit_left[0] is not None and limit_left[0] <= 0:
                            if bar is not None and hasattr(bar, "close"):
                                bar.close()
                            return
                        lim = 0 if limit_left[0] is None else limit_left[0]
                    n = run_modality(
                        mini,
                        client,
                        model_name,
                        image_paths,
                        pred_field,
                        args.overwrite,
                        lim,
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

    if args.modality == "both":
        out_one = (
            Path(args.output_json).expanduser().resolve()
            if args.output_json
            else default_benchmark_merged_output_path(benchmark_path, args.model)
        )
        preds_jsonl = resolve_benchmark_preds_jsonl_path(
            out_one, getattr(args, "predictions_jsonl", "") or ""
        )
        pred_fields_both = ["pred_answer_multiview", "pred_answer_walkthrough"]

        if args.overwrite and preds_jsonl.is_file():
            preds_jsonl.unlink()

        # Questions always follow the current --questions-json (payload).
        work = json.loads(json.dumps(payload))

        if (
            not args.overwrite
            and not preds_jsonl.is_file()
            and out_one.is_file()
        ):
            n_mig = export_benchmark_preds_to_jsonl(out_one, preds_jsonl, pred_fields_both, conditions)
            print(f"[INFO] Migrated {n_mig} predictions from existing eval JSON -> {preds_jsonl}")

        preds_map = load_preds_jsonl_map(preds_jsonl)
        merge_preds_map_into_benchmark_work(work, preds_map, conditions, pred_fields_both)

        if rerun_types:
            cleared, cleared_ids = _clear_benchmark_predictions_for_types(
                work,
                pred_fields_both,
                rerun_types,
                conditions,
            )
            print(f"[INFO] Cleared predictions for rerun types {sorted(rerun_types)}: {cleared} field(s)")
            rm = filter_jsonl_drop_question_ids(preds_jsonl, cleared_ids)
            print(f"[INFO] Removed {rm} JSONL lines for those question_ids")

        already_mv = _count_existing_predictions(work, "pred_answer_multiview")
        already_wt = _count_existing_predictions(work, "pred_answer_walkthrough")
        already_total = already_mv + already_wt
        print(
            f"[INFO] Benchmark (questions from input); preds sidecar: {preds_jsonl} "
            f"(already={already_total}) merged eval -> {out_one}"
        )

        try:
            from tqdm.auto import tqdm

            computed_rem = _count_benchmark_total(
                "multiview", "pred_answer_multiview", work
            ) + _count_benchmark_total("walkthrough", "pred_answer_walkthrough", work)
            total_for_bar = (args.limit if args.limit > 0 else computed_rem) or 0
            if total_for_bar > 0:
                bar = tqdm(total=total_for_bar, desc="benchmark", unit="q")
                if args.limit > 0:
                    try:
                        bar.update(min(already_total, args.limit))
                    except Exception:
                        pass
        except Exception:
            bar = None

        # --limit means: total desired predictions across BOTH fields in the merged output.
        if args.limit > 0:
            remaining = max(0, args.limit - already_total)
            limit_state = [remaining]
        else:
            limit_state = [None]

        if limit_state[0] == 0:
            print(f"Skip benchmark (limit already satisfied): {out_one}")
            if bar is not None and hasattr(bar, "close"):
                bar.close()
            return
        print("\n=== Occlusion benchmark | multiview → pred_answer_multiview ===")
        fill_payload(
            work,
            "multiview",
            pred_field="pred_answer_multiview",
            limit_left=limit_state,
            preds_jsonl_path=preds_jsonl,
        )
        print("\n=== Occlusion benchmark | walkthrough → pred_answer_walkthrough ===")
        fill_payload(
            work,
            "walkthrough",
            pred_field="pred_answer_walkthrough",
            limit_left=limit_state,
            preds_jsonl_path=preds_jsonl,
        )
        save_json(out_one, work)
        print(f"\nSaved (single file): {out_one}")
        print_benchmark_report(evaluate_benchmark_predictions(work, "pred_answer_multiview"))
        print_benchmark_report(evaluate_benchmark_predictions(work, "pred_answer_walkthrough"))
        if bar is not None and hasattr(bar, "close"):
            bar.close()
        return

    out_path = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else default_benchmark_output_path(benchmark_path, args.model, args.modality)
    )
    preds_jsonl = resolve_benchmark_preds_jsonl_path(
        out_path, getattr(args, "predictions_jsonl", "") or ""
    )

    if args.overwrite and preds_jsonl.is_file():
        preds_jsonl.unlink()

    work = json.loads(json.dumps(payload))

    if (
        not args.overwrite
        and not preds_jsonl.is_file()
        and out_path.is_file()
    ):
        n_mig = export_benchmark_preds_to_jsonl(out_path, preds_jsonl, [PRED_FIELD], conditions)
        print(f"[INFO] Migrated {n_mig} predictions from existing eval JSON -> {preds_jsonl}")

    preds_map = load_preds_jsonl_map(preds_jsonl)
    merge_preds_map_into_benchmark_work(work, preds_map, conditions, [PRED_FIELD])

    if rerun_types:
        cleared, cleared_ids = _clear_benchmark_predictions_for_types(
            work,
            [PRED_FIELD],
            rerun_types,
            conditions,
        )
        print(f"[INFO] Cleared predictions for rerun types {sorted(rerun_types)}: {cleared} field(s)")
        rm = filter_jsonl_drop_question_ids(preds_jsonl, cleared_ids)
        print(f"[INFO] Removed {rm} JSONL lines for those question_ids")

    already = _count_existing_predictions(work, PRED_FIELD)
    print(
        f"[INFO] Benchmark (questions from input); preds sidecar: {preds_jsonl} "
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

    # --limit means: total desired predictions in the output file (resume up to N).
    if args.limit > 0:
        remaining = max(0, args.limit - already)
        limit_state = [remaining]
    else:
        limit_state = [None]

    if limit_state[0] == 0:
        print(f"Skip benchmark (limit already satisfied): {out_path}")
        if bar is not None and hasattr(bar, "close"):
            bar.close()
        return
    if args.modality == "text":
        print("Occlusion benchmark | text-only (no images sent to model).")
        fill_payload(
            work,
            "text",
            pred_field=PRED_FIELD,
            limit_left=limit_state,
            preds_jsonl_path=preds_jsonl,
        )
    elif args.modality == "singleview":
        print("Occlusion benchmark | singleview (oracle frame image per block).")
        fill_payload(
            work,
            "singleview",
            pred_field=PRED_FIELD,
            limit_left=limit_state,
            preds_jsonl_path=preds_jsonl,
        )
    elif args.modality == "multiview":
        fill_payload(
            work,
            "multiview",
            pred_field=PRED_FIELD,
            limit_left=limit_state,
            preds_jsonl_path=preds_jsonl,
        )
    else:
        fill_payload(
            work,
            "walkthrough",
            pred_field=PRED_FIELD,
            limit_left=limit_state,
            preds_jsonl_path=preds_jsonl,
        )

    save_json(out_path, work)
    print(f"\nSaved: {out_path}")
    print_benchmark_report(evaluate_benchmark_predictions(work, PRED_FIELD))
    if bar is not None and hasattr(bar, "close"):
        bar.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Vision MC eval for Azure/OpenAI. Occlusion: use occlusion_benchmark.json as --questions-json. "
            "Optional: one target_questions.json + --multiview-dir + --video-path (non-benchmark)."
        ),
    )
    parser.add_argument(
        "--questions-json",
        default="/nas-ssd2/yuezhang/Holodeck/3d_essential/generated_scenes/occlusion_benchmark.json",
        help=(
            "occlusion_benchmark.json (scenes → targets → clean | full_occlusion + image_retrieval), "
            "or a single target_questions.json with records[].questions[]."
        ),
    )
    parser.add_argument(
        "--multiview-dir",
        default="/nas-ssd2/yuezhang/Holodeck/3d_essential/occlusion_scenes/a_bedroom-2026-02-02-15-20-11-693081/bookshelf-0_bedroom/occlusion_bookshelf-0_bedroom_wardrobe-0_bedroom/multiview"
    )
    parser.add_argument(
        "--video-path",
        default="/nas-ssd2/yuezhang/Holodeck/3d_essential/occlusion_scenes/a_bedroom-2026-02-02-15-20-11-693081/bookshelf-0_bedroom/occlusion_bookshelf-0_bedroom_wardrobe-0_bedroom/walkthrough.mp4"
    )
    parser.add_argument("--provider", default="auto", choices=["auto", "openai", "azure"])
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--video-frames", type=int, default=8)
    parser.add_argument(
        "--modality",
        default="multiview",
        choices=["multiview", "singleview", "walkthrough", "both", "text"],
        help="Choose one modality (multiview/singleview/walkthrough/text) or run both visual (multiview+walkthrough). 'text' sends no images.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help=(
            "Optional output path. Single-scene: default next to input. "
            "Occlusion benchmark: default next to benchmark json; "
            "--modality both writes ONE merged file here (pred_answer_multiview + pred_answer_walkthrough)."
        ),
    )
    parser.add_argument(
        "--predictions-jsonl",
        dest="predictions_jsonl",
        default="",
        help=(
            "Append-only predictions log (JSONL), one object per line: "
            '{ "question_id": "...", "pred_field": "pred_answer", "pred": "A" }. '
            "Default for benchmark: <output_stem>_preds.jsonl next to the eval JSON."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Occlusion benchmark: total desired predictions in the output JSON (resume up to N; 0 = all). "
            "Non-benchmark: max new predictions to write this run (0 = all)."
        ),
    )
    parser.add_argument(
        "--rerun-question-types",
        dest="rerun_question_types",
        default="",
        help=(
            "Comma-separated question_type values to re-evaluate even if pred_* already exists "
            "(benchmark mode only). Example: --rerun-question-types relative_position"
        ),
    )
    parser.add_argument(
        "--benchmark-condition",
        default="both",
        choices=["both", "clean", "partial_occlusion", "full_occlusion"],
        help="When input is occlusion_benchmark.json: run clean, partial_occlusion, full_occlusion block, or both (all).",
    )
    parser.add_argument(
        "--prompt-style",
        dest="prompt_style",
        default="default",
        choices=["default", "cot"],
        help=(
            "Prompt strategy: 'default' = direct MC answer; "
            "'cot' = chain-of-thought reasoning before answering. "
            "Non-default styles are appended to the output filename."
        ),
    )
    args = parser.parse_args()

    # Append prompt style to output filename when non-default
    if args.prompt_style != "default" and not args.output_json:
        # Patch default_benchmark_output_path / default_output_path by injecting style into model name
        args._model_for_output = f"{args.model}_{args.prompt_style}"
    else:
        args._model_for_output = args.model

    client, model_name, backend = make_client(args.provider, args.model)
    print(f"Backend: {backend}, model: {model_name}")

    def _deepcopy_payload(obj: dict[str, Any]) -> dict[str, Any]:
        return json.loads(json.dumps(obj))

    def _run_save_eval(payload_to_use: dict[str, Any], images: list[Path], out_path: Path) -> None:
        rerun_types = _parse_question_types_csv(getattr(args, "rerun_question_types", "") or "")
        pj: Path | None = None
        if (getattr(args, "predictions_jsonl", "") or "").strip():
            pj = resolve_benchmark_preds_jsonl_path(out_path, args.predictions_jsonl)
        run_modality(
            payload=payload_to_use,
            client=client,
            model_name=model_name,
            image_paths=images,
            pred_field=PRED_FIELD,
            overwrite=args.overwrite,
            limit=args.limit,
            rerun_question_types=rerun_types,
            preds_jsonl_path=pj,
            prompt_style=getattr(args, "prompt_style", "default"),
        )
        save_json(out_path, payload_to_use)
        print(f"\nSaved predictions to: {out_path}")
        report = evaluate(payload_to_use, PRED_FIELD)
        print_report(report)

    def _process_scene(q_path: Path, mv_dir: Path, video_path: Path, allow_custom_output: bool) -> None:
        if not q_path.is_file():
            raise FileNotFoundError(f"questions json not found: {q_path}")
        if args.modality not in {"text"} and not mv_dir.is_dir():
            raise FileNotFoundError(f"multiview dir not found: {mv_dir}")
        if args.modality in {"walkthrough", "both"} and not video_path.is_file():
            raise FileNotFoundError(f"video not found: {video_path}")

        payload = load_json(q_path)

        mv_images = list_images(mv_dir, args.max_images)
        if args.modality in {"multiview", "singleview", "both"}:
            if not mv_images:
                raise RuntimeError(f"No images found in {mv_dir}")
            print(f"Loaded {len(mv_images)} multiview images.")

        video_frames: list[Path] = []
        if args.modality in {"walkthrough", "both"}:
            video_frames = sample_video_frames(video_path, args.video_frames)
            print(f"Extracted {len(video_frames)} walkthrough frames.")

        if args.modality == "both":
            payload_mv = _deepcopy_payload(payload)
            out_mv = default_output_path(q_path, args.model, "multiview")
            _run_save_eval(payload_mv, mv_images, out_mv)

            payload_wt = _deepcopy_payload(payload)
            out_wt = default_output_path(q_path, args.model, "walkthrough")
            _run_save_eval(payload_wt, video_frames, out_wt)
            return

        # Single modality (text / multiview / singleview / walkthrough)
        if args.modality == "text":
            print("Text-only mode: no images sent to model.")
            images = []
        elif args.modality in {"multiview", "singleview"}:
            if args.modality == "singleview":
                # Non-benchmark mode doesn't have per-block oracle metadata.
                # Use the first multiview image as the singleview proxy.
                single = [mv_images[0]] if mv_images else []
                print(f"Singleview image selected: {single[0] if single else 'none'}")
                images = single
            else:
                images = mv_images
        else:
            images = video_frames

        if allow_custom_output and args.output_json:
            out_path = Path(args.output_json).resolve()
        else:
            out_path = default_output_path(q_path, args.model, args.modality)
        _run_save_eval(payload, images, out_path)

    q_path = Path(args.questions_json).expanduser().resolve()
    if q_path.is_file():
        bench_payload = load_json(q_path)
        if is_occlusion_benchmark(bench_payload):
            process_occlusion_benchmark(q_path, bench_payload, args, client, model_name)
            return

    # Single JSON + explicit multiview dir + video (non-benchmark)
    mv_dir = Path(args.multiview_dir).resolve()
    video_path = Path(args.video_path).resolve()
    _process_scene(q_path, mv_dir, video_path, allow_custom_output=True)


if __name__ == "__main__":
    main()
