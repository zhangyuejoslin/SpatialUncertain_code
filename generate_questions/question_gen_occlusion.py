from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any


CANNOT_DETERMINE = "Cannot determine from this viewpoint"

# Scene types where the target is (at least partially) visible — GT is computed from 3D data.
# full_occlusion → target invisible → spatial answers become CANNOT_DETERMINE.
_VISIBLE_SCENE_TYPES = {"clean", "no_occlude", "partial_occlude"}


# ──────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _format_path_for_json(path: Path | str | None, anchor: Path | None) -> str | None:
    """If `anchor` is set, store paths relative to it (posix-style); else absolute."""
    if path is None:
        return None
    p = Path(path).expanduser().resolve()
    if anchor is None:
        return str(p)
    a = anchor.expanduser().resolve()
    rel = os.path.relpath(p, a)
    return rel.replace("\\", "/")


# ──────────────────────────────────────────────
# Math
# ──────────────────────────────────────────────

def _vec(p: dict[str, Any]) -> tuple[float, float, float]:
    return (float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0)))


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(v, v))


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    n = _norm(v)
    if n < 1e-8:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _camera_basis(rotation: dict[str, Any]) -> dict[str, tuple[float, float, float]]:
    rx = math.radians(float(rotation.get("x", 0.0)))
    ry = math.radians(float(rotation.get("y", 0.0)))
    forward = _normalize((math.sin(ry) * math.cos(rx), -math.sin(rx), math.cos(ry) * math.cos(rx)))
    right = _normalize(_cross((0.0, 1.0, 0.0), forward))
    if _norm(right) < 1e-8:
        right = (1.0, 0.0, 0.0)
    up = _normalize(_cross(forward, right))
    return {"forward": forward, "right": right, "up": up}


# ──────────────────────────────────────────────
# Name helpers
# ──────────────────────────────────────────────

def _question_name(name: str) -> str:
    base = re.sub(r"\s*\([^)]*\)\s*$", "", str(name)).strip()
    base = re.sub(r"[_-]+", " ", base).strip()
    return re.sub(r"\s+", " ", base).strip()


def _human_object_name(name: str) -> str:
    cleaned = _question_name(name)
    cleaned = re.sub(r"\b\d+\b$", "", cleaned).strip()
    return re.sub(r"-\d+$", "", cleaned).strip()


def _tokenize_id(name: str) -> str:
    raw = str(name).strip()
    # Some exports use '|' to separate object and parent/container parts, while other
    # files (e.g. occlusion_meta / folder names) use '_' as separator.
    raw = raw.replace("|", "_")
    if "(" in raw and raw.endswith(")"):
        base, room = raw.rsplit("(", 1)
        room = room[:-1].strip()
        token = f"{base.strip().replace(' ', '_')}_{room.replace(' ', '_')}"
    else:
        token = raw.replace(" ", "_")
    # Collapse repeated underscores from mixed separators/spaces.
    token = re.sub(r"_+", "_", token)
    return token.strip("_")


# ──────────────────────────────────────────────
# MC helpers
# ──────────────────────────────────────────────

def _shuffle_options(options: list[str], answer_text: str, seed_text: str) -> tuple[list[str], str]:
    shuffled = list(options)
    random.Random(seed_text).shuffle(shuffled)
    letters = ["A", "B", "C", "D", "E"]
    labeled = [f"{letters[i]}) {opt}" for i, opt in enumerate(shuffled)]
    gt_letter = letters[shuffled.index(answer_text)]
    return labeled, gt_letter


# ──────────────────────────────────────────────
# Object geometry
# ──────────────────────────────────────────────

def _aabb_center(obj: dict[str, Any]) -> tuple[float, float, float]:
    aabb = obj.get("aabb") or {}
    center = aabb.get("center") or obj.get("position") or {}
    return _vec(center)


def _aabb_volume(obj: dict[str, Any]) -> float:
    aabb = obj.get("aabb") or {}
    if "size" in aabb:
        s = aabb["size"]
        return (abs(float(s.get("x", 1.0)))
                * abs(float(s.get("y", 1.0)))
                * abs(float(s.get("z", 1.0))))
    if "min" in aabb and "max" in aabb:
        mn, mx = aabb["min"], aabb["max"]
        return (abs(float(mx.get("x", 0)) - float(mn.get("x", 0)))
                * abs(float(mx.get("y", 0)) - float(mn.get("y", 0)))
                * abs(float(mx.get("z", 0)) - float(mn.get("z", 0))))
    return 1.0


# ──────────────────────────────────────────────
# Scene loading
# ──────────────────────────────────────────────

def _parse_scene_folder(input_path: Path) -> Path:
    if input_path.is_dir():
        return input_path
    if input_path.name in {"target_subgraph.json", "occlusion_meta.json"}:
        return input_path.parent
    raise ValueError("Input must be a scene folder or a metadata file inside it.")


def _occlusion_subdir_for_pair(scene_dir: Path, target_raw: str, occluder_raw: str) -> Path | None:
    """
    If scene_dir is the clean scene root (parent of occlusion_* folders), return the
    occlusion_* child that matches target/occluder tokens (same naming as folders).
    """
    token_t = _tokenize_id(target_raw)
    token_o = _tokenize_id(occluder_raw)
    name = f"occlusion_{token_t}_{token_o}"
    cand = scene_dir / name
    if cand.is_dir() and (cand / "object_attributes.json").exists():
        return cand
    return None


def _default_single_output_dir(
    scene_dir: Path,
    scene_type: str,
    target_id: str | None,
    occluder_id: str | None,
) -> Path:
    """
    Put questions_clean.json and questions_full_occlusion.json in the same folder:
    always the occlusion_* directory when we can resolve it.
    """
    if scene_dir.name.startswith("occlusion_"):
        return scene_dir
    if scene_type == "clean" and target_id and occluder_id:
        occ = _occlusion_subdir_for_pair(scene_dir, target_id, occluder_id)
        if occ is not None:
            return occ
    return scene_dir


def _load_scene_assets(
    scene_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    object_path = scene_dir / "object_attributes.json"
    camera_path = scene_dir / "camera_poses.json"
    meta_path = scene_dir / "occlusion_meta.json"
    if not object_path.exists():
        raise FileNotFoundError(f"Missing object_attributes.json: {object_path}")
    if not camera_path.exists():
        raise FileNotFoundError(f"Missing camera_poses.json: {camera_path}")
    objects = load_json(object_path)
    camera_data = load_json(camera_path)
    meta = load_json(meta_path) if meta_path.exists() else None
    if not isinstance(objects, list):
        raise ValueError(f"Invalid object_attributes.json: {object_path}")
    if not isinstance(camera_data, dict):
        raise ValueError(f"Invalid camera_poses.json: {camera_path}")
    return objects, camera_data, meta


def _normalize_object_id(raw_id: str, object_map: dict[str, dict[str, Any]]) -> str:
    raw = str(raw_id).strip()
    if raw in object_map:
        return raw
    token = _tokenize_id(raw)
    matches = [oid for oid in object_map if _tokenize_id(oid) == token]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Could not resolve object id: {raw_id!r}")


def _resolve_target_and_occluder(
    scene_dir: Path,
    object_map: dict[str, dict[str, Any]],
    meta: dict[str, Any] | None,
) -> tuple[str, str]:
    if meta is not None:
        target_raw = meta.get("target_id") or meta.get("target_token")
        occluder_raw = meta.get("occluder_id") or meta.get("occluder_token")
        if target_raw and occluder_raw:
            return (
                _normalize_object_id(str(target_raw), object_map),
                _normalize_object_id(str(occluder_raw), object_map),
            )
        raise ValueError(f"occlusion_meta.json missing target/occluder: {scene_dir / 'occlusion_meta.json'}")
    folder = scene_dir.name
    if not folder.startswith("occlusion_"):
        raise ValueError(f"Cannot infer target/occluder from folder name: {folder}")
    body = folder[len("occlusion_"):]
    candidates = [(_tokenize_id(oid), oid) for oid in object_map]
    for target_token, target_id in candidates:
        prefix = target_token + "_"
        if not body.startswith(prefix):
            continue
        occluder_token = body[len(prefix):]
        for cand_token, cand_id in candidates:
            if cand_token == occluder_token:
                return target_id, cand_id
    raise ValueError(f"Could not parse target/occluder from folder: {folder}")


def _get_frame(
    camera_data: dict[str, Any],
    meta: dict[str, Any] | None = None,
    frame_idx: int | None = None,
) -> tuple[str, dict[str, Any]]:
    seq = (camera_data.get("camera_frame_for_modality_eval")
           or camera_data.get("primary_sequence")
           or "multiview")
    frames = ((camera_data.get(seq) or {}).get("frames") or [])
    if not isinstance(frames, list) or not frames:
        raise ValueError("camera_poses.json has no valid frame list")

    # Priority: explicit frame_idx > oracle_frame in meta > frame 0
    target_idx = frame_idx
    if target_idx is None and meta:
        oracle = meta.get("oracle_frame") or {}
        if isinstance(oracle, dict) and "frame_idx" in oracle:
            target_idx = int(oracle["frame_idx"])

    if target_idx is not None:
        frame = next(
            (f for f in frames if isinstance(f, dict) and int(f.get("frame_idx", -1)) == target_idx),
            None,
        )
        if frame is not None:
            return str(seq), frame

    frame0 = frames[0]
    if not isinstance(frame0, dict):
        raise ValueError("First frame in camera_poses.json is invalid")
    return str(seq), frame0


# ──────────────────────────────────────────────
# GT computation (3D ground truth for clean scenes)
# ──────────────────────────────────────────────

def _compute_relative_direction(
    target_id: str,
    referent_id: str,
    object_map: dict[str, dict[str, Any]],
    frame: dict[str, Any],
) -> str:
    """Return the dominant 2D direction of target relative to referent in camera view.

    Computes horizontal (left/right) and vertical (above/below) offset in camera space,
    then picks whichever axis has the larger magnitude. If both are close, combines them.
    """
    cam_pos = _vec(frame.get("position", {}))
    basis = _camera_basis(frame.get("rotation", {}))

    t_center = _aabb_center(object_map[target_id])
    r_center = _aabb_center(object_map[referent_id])
    diff = _sub(t_center, r_center)

    dx = _dot(diff, basis["right"])   # positive = target is to the right
    dy = _dot(diff, basis["up"])      # positive = target is above

    h = "left" if dx < 0 else "right"
    v = "above" if dy > 0 else "below"

    abs_dx, abs_dy = abs(dx), abs(dy)
    # If one axis dominates (>2x the other), use only that axis
    if abs_dx > abs_dy * 2:
        return h
    if abs_dy > abs_dx * 2:
        return v
    return f"{v}-{h}"  # e.g. "above-left", "below-right"


def _compute_depth_ordering(
    target_id: str,
    occluder_id: str,
    object_map: dict[str, dict[str, Any]],
    frame: dict[str, Any],
) -> str:
    """'target' or 'occluder' — whichever is closer to camera."""
    cam_pos = _vec(frame.get("position", {}))
    basis = _camera_basis(frame.get("rotation", {}))
    tz = _dot(_sub(_aabb_center(object_map[target_id]), cam_pos), basis["forward"])
    oz = _dot(_sub(_aabb_center(object_map[occluder_id]), cam_pos), basis["forward"])
    return "target" if tz < oz else "occluder"


def _compute_size_relation(
    target_id: str,
    occluder_id: str,
    object_map: dict[str, dict[str, Any]],
) -> str:
    """'larger', 'smaller', or 'similar'."""
    tv = _aabb_volume(object_map[target_id])
    ov = _aabb_volume(object_map[occluder_id])
    ratio = tv / ov if ov > 1e-6 else 1.0
    if ratio > 1.3:
        return "larger"
    if ratio < 1.0 / 1.3:
        return "smaller"
    return "similar"


# ──────────────────────────────────────────────
# Referent selection
# ──────────────────────────────────────────────

def _is_in_frustum(
    obj: dict[str, Any],
    frame: dict[str, Any],
    fov_h_deg: float = 90.0,
    aspect: float = 778 / 497,
) -> bool:
    """Return True if the object's AABB center is inside the camera frustum."""
    cam_pos = _vec(frame.get("position", {}))
    basis = _camera_basis(frame.get("rotation", {}))
    fov_h = math.radians(fov_h_deg)
    fov_v = 2 * math.atan(math.tan(fov_h / 2) / aspect)

    center = _aabb_center(obj)
    diff = _sub(center, cam_pos)
    fwd = _dot(diff, basis["forward"])
    if fwd <= 0:
        return False  # behind camera
    rgt = _dot(diff, basis["right"])
    up  = _dot(diff, basis["up"])
    return abs(math.atan2(rgt, fwd)) < fov_h / 2 and abs(math.atan2(up, fwd)) < fov_v / 2


def _pick_referent(
    target_id: str,
    occluder_id: str,
    object_map: dict[str, dict[str, Any]],
    frame: dict[str, Any] | None = None,
) -> str | None:
    """Pick a deterministic random object that is:
    - not the target or occluder
    - visible in the oracle frame (frustum check), falling back to all others if none qualify
    """
    exclude = {target_id, occluder_id}
    candidates = sorted(oid for oid in object_map if oid and oid not in exclude)
    if not candidates:
        return None

    if frame is not None:
        visible = [oid for oid in candidates if _is_in_frustum(object_map[oid], frame)]
        if visible:
            candidates = visible

    return random.Random(target_id).choice(candidates)


def _pick_size_referent(
    target_id: str,
    occluder_id: str,
    object_map: dict[str, dict[str, Any]],
    frame: dict[str, Any] | None = None,
) -> str | None:
    """Pick a referent for the size comparison question.

    Prefers an object whose volume is within 3x of the target's volume
    (non-obvious from common sense). Falls back to any visible object,
    then any non-target/occluder object.
    """
    exclude = {target_id, occluder_id}
    candidates = sorted(oid for oid in object_map if oid and oid not in exclude)
    if not candidates:
        return None

    if frame is not None:
        visible = [oid for oid in candidates if _is_in_frustum(object_map[oid], frame)]
        if visible:
            candidates = visible

    target_vol = _aabb_volume(object_map[target_id])
    if target_vol > 1e-6:
        # Prefer objects with a clear size difference (ratio > 1.3), so the answer is visually obvious
        distinguishable = [
            oid for oid in candidates
            if not (1/1.3 <= _aabb_volume(object_map[oid]) / target_vol <= 1.3)
        ]
        if distinguishable:
            candidates = distinguishable

    return random.Random(f"{target_id}|size").choice(candidates)


# ──────────────────────────────────────────────
# Question builders
# Each question: clean/partial → deterministic GT from 3D, full_occlusion → CANNOT_DETERMINE
# Q1 (visibility): asks about target only (occluder is occlusion context, not referent)
# Q2–Q4 (spatial): use referent — a random object that is NOT the target or occluder
# ──────────────────────────────────────────────

def _build_visibility_question(
    target_name: str,
    scene_type: str,
) -> dict[str, Any]:
    answer_text = "Yes" if scene_type in _VISIBLE_SCENE_TYPES else "No"
    options = ["Yes", "No"]
    labeled, gt_letter = _shuffle_options(
        options, answer_text, f"{target_name}|vis|{scene_type}"
    )
    return {
        "question": f"From this camera viewpoint, is the {target_name} visible?",
        "question_type": "visibility",
        "answer_text": answer_text,
        "answer_options": labeled,
        "gt_answer": gt_letter,
        "scene_type": scene_type,
    }


def _build_relative_position_question(
    target_name: str,
    referent_name: str,
    target_id: str,
    referent_id: str,
    object_map: dict[str, dict[str, Any]],
    frame: dict[str, Any],
    scene_type: str,
) -> dict[str, Any]:
    options = [
        f"To the left of the {referent_name}",
        f"To the right of the {referent_name}",
        f"Above the {referent_name}",
        f"Below the {referent_name}",
        CANNOT_DETERMINE,
    ]
    if scene_type in _VISIBLE_SCENE_TYPES:
        direction = _compute_relative_direction(target_id, referent_id, object_map, frame)
        # Map combined directions (e.g. "above-left") to the dominant axis option
        if "left" in direction and "above" not in direction and "below" not in direction:
            answer_text = f"To the left of the {referent_name}"
        elif "right" in direction and "above" not in direction and "below" not in direction:
            answer_text = f"To the right of the {referent_name}"
        elif direction.startswith("above"):
            answer_text = f"Above the {referent_name}"
        elif direction.startswith("below"):
            answer_text = f"Below the {referent_name}"
        else:
            answer_text = f"To the left of the {referent_name}" if "left" in direction else f"To the right of the {referent_name}"
    else:
        answer_text = CANNOT_DETERMINE
    labeled, gt_letter = _shuffle_options(
        options, answer_text, f"{target_id}|{referent_id}|rel_pos|{scene_type}"
    )
    return {
        "question": (
            f"From this camera viewpoint, where is the {target_name} "
            f"relative to the {referent_name}?"
        ),
        "question_type": "relative_position",
        "answer_text": answer_text,
        "answer_options": labeled,
        "gt_answer": gt_letter,
        "scene_type": scene_type,
    }


def _build_depth_ordering_question(
    target_name: str,
    referent_name: str,
    target_id: str,
    referent_id: str,
    object_map: dict[str, dict[str, Any]],
    frame: dict[str, Any],
    scene_type: str,
) -> dict[str, Any]:
    options = [
        f"The {target_name}",
        f"The {referent_name}",
        "They are at equal distance",
        CANNOT_DETERMINE,
    ]
    if scene_type in _VISIBLE_SCENE_TYPES:
        closer = _compute_depth_ordering(target_id, referent_id, object_map, frame)
        answer_text = f"The {target_name}" if closer == "target" else f"The {referent_name}"
    else:
        answer_text = CANNOT_DETERMINE
    labeled, gt_letter = _shuffle_options(
        options, answer_text, f"{target_id}|{referent_id}|depth|{scene_type}"
    )
    return {
        "question": (
            f"Which object is closer to the camera: "
            f"the {target_name} or the {referent_name}?"
        ),
        "question_type": "depth_ordering",
        "answer_text": answer_text,
        "answer_options": labeled,
        "gt_answer": gt_letter,
        "scene_type": scene_type,
    }


def _build_size_question(
    target_name: str,
    referent_name: str,
    target_id: str,
    referent_id: str,
    object_map: dict[str, dict[str, Any]],
    scene_type: str,
) -> dict[str, Any]:
    options = [
        f"The {target_name} is larger",
        f"The {target_name} is smaller",
        "They are about the same size",
        CANNOT_DETERMINE,
    ]
    if scene_type in _VISIBLE_SCENE_TYPES:
        rel = _compute_size_relation(target_id, referent_id, object_map)
        if rel == "larger":
            answer_text = f"The {target_name} is larger"
        else:
            answer_text = f"The {target_name} is smaller"
    else:
        answer_text = CANNOT_DETERMINE
    labeled, gt_letter = _shuffle_options(
        options, answer_text, f"{target_id}|{referent_id}|size|{scene_type}"
    )
    return {
        "question": (
            f"Compared to the {referent_name}, "
            f"is the {target_name} larger or smaller?"
        ),
        "question_type": "size_comparison",
        "answer_text": answer_text,
        "answer_options": labeled,
        "gt_answer": gt_letter,
        "scene_type": scene_type,
    }


# ──────────────────────────────────────────────
# Main generation
# ──────────────────────────────────────────────

def generate_questions(
    scene_dir: Path,
    scene_type: str,                        # "clean" | "full_occlusion"
    target_id_override: str | None = None,
    occluder_id_override: str | None = None,
    oracle_frame_idx: int | None = None,
    path_anchor: Path | None = None,
) -> dict[str, Any]:
    objects, camera_data, meta = _load_scene_assets(scene_dir)
    object_map = {str(o.get("id") or o.get("thor_objectId") or ""): o for o in objects}

    if target_id_override and occluder_id_override:
        target_id = _normalize_object_id(target_id_override, object_map)
        occluder_id = _normalize_object_id(occluder_id_override, object_map)
    else:
        target_id, occluder_id = _resolve_target_and_occluder(scene_dir, object_map, meta)

    sequence_name, frame = _get_frame(camera_data, meta, oracle_frame_idx)
    target_name = _human_object_name(target_id)

    # Referent for spatial questions (relative position, depth): any visible non-target/occluder
    referent_id = _pick_referent(target_id, occluder_id, object_map, frame)
    if referent_id is None:
        raise ValueError(f"No referent object available in scene: {scene_dir}")
    referent_name = _human_object_name(referent_id)

    # Referent for size question: prefer similar-sized object to avoid common-sense trivial answers
    size_referent_id = _pick_size_referent(target_id, occluder_id, object_map, frame)
    if size_referent_id is None:
        size_referent_id = referent_id
    size_referent_name = _human_object_name(size_referent_id)

    questions = [
        _build_visibility_question(target_name, scene_type),
        _build_relative_position_question(
            target_name, referent_name, target_id, referent_id, object_map, frame, scene_type
        ),
        _build_depth_ordering_question(
            target_name, referent_name, target_id, referent_id, object_map, frame, scene_type
        ),
        _build_size_question(
            target_name, size_referent_name, target_id, size_referent_id, object_map, scene_type
        ),
    ]

    # Assign stable question IDs: scene/target/condition/question_type
    scene_slug = scene_dir.name
    for q in questions:
        q["question_id"] = f"{scene_slug}|{target_id}|{scene_type}|{q['question_type']}"

    return {
        "meta": {
            "scene_dir": _format_path_for_json(scene_dir, path_anchor),
            "scene_type": scene_type,
            "target_id": target_id,
            "target_name": target_name,
            "occluder_id": occluder_id,
            "referent_id": referent_id,
            "referent_name": referent_name,
            "size_referent_id": size_referent_id,
            "size_referent_name": size_referent_name,
            "camera_sequence": sequence_name,
            "frame_idx": frame.get("frame_idx"),
        },
        "questions": questions,
    }


# ──────────────────────────────────────────────
# Image / video retrieval
# ──────────────────────────────────────────────

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _resolve_oracle_image(scene_dir: Path, sequence: str, frame_idx: int, oracle_meta: dict) -> str | None:
    """Try to find the image file for the oracle frame."""
    # 1. Use image_file from oracle_meta if available
    img_file = oracle_meta.get("image_file") or oracle_meta.get("image_path")
    if img_file:
        p = scene_dir / img_file
        if p.exists():
            return str(p)

    # 2. Try common naming patterns
    for pat in [
        f"{sequence}/{frame_idx:03d}.jpg",
        f"{sequence}/{frame_idx:03d}.png",
        f"{sequence}/view_{frame_idx:03d}.jpg",
        f"{sequence}/view_{frame_idx:03d}.png",
        f"{sequence}/frame_{frame_idx:03d}.jpg",
        f"{sequence}/frame_{frame_idx:03d}.png",
    ]:
        p = scene_dir / pat
        if p.exists():
            return str(p)
    return None


def _resolve_sequence_images(scene_dir: Path, sequence: str) -> list[str]:
    """Return sorted list of all image paths in the sequence folder."""
    seq_dir = scene_dir / sequence
    if not seq_dir.is_dir():
        return []
    return sorted(str(p) for p in seq_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)


def _build_image_retrieval(
    scene_dir: Path,
    camera_data: dict[str, Any],
    oracle_frame_idx: int | None,
    oracle_meta: dict | None = None,
    path_anchor: Path | None = None,
) -> dict[str, Any]:
    seq = (
        camera_data.get("camera_frame_for_modality_eval")
        or camera_data.get("primary_sequence")
        or "multiview"
    )
    oracle_meta = oracle_meta or {}
    oracle_image = (
        _resolve_oracle_image(scene_dir, seq, oracle_frame_idx, oracle_meta)
        if oracle_frame_idx is not None
        else None
    )
    all_images = _resolve_sequence_images(scene_dir, seq)
    return {
        "sequence": seq,
        "oracle_frame": {
            "frame_idx": oracle_frame_idx,
            "image_path": _format_path_for_json(oracle_image, path_anchor),
        },
        "multiview": {
            "frame_count": len(all_images),
            "image_paths": [_format_path_for_json(p, path_anchor) for p in all_images],
        },
    }


# ──────────────────────────────────────────────
# Eval unit builder
# ──────────────────────────────────────────────

def _occ_condition_key(scene_type: str) -> str:
    """Map internal scene_type to the condition key used in eval_unit conditions dict."""
    return "partial_occlusion" if scene_type == "partial_occlude" else "full_occlusion"


def _make_unit_id(scene_name: str, target_id: str, occluder_id: str) -> str:
    return f"{scene_name}__{_tokenize_id(target_id)}__{_tokenize_id(occluder_id)}"


def build_eval_unit(
    occlusion_dir: Path,
    clean_dir: Path,
    payload_occ: dict[str, Any],
    payload_clean: dict[str, Any],
    target_id: str,
    occluder_id: str,
    oracle_frame_idx: int | None,
    camera_data_occ: dict[str, Any],
    camera_data_clean: dict[str, Any],
    meta_occ: dict[str, Any] | None,
    path_anchor: Path | None = None,
    occlusion_scene_type: str = "full_occlusion",
) -> dict[str, Any]:
    target_name = _human_object_name(target_id)
    occluder_name = _human_object_name(occluder_id)
    scene_name = clean_dir.name
    unit_id = _make_unit_id(scene_name, target_id, occluder_id)

    oracle_meta = (meta_occ.get("oracle_frame") or {}) if meta_occ else {}

    return {
        "unit_id": unit_id,
        "scene_name": scene_name,
        "target_id": target_id,
        "target_name": target_name,
        "occluder_id": occluder_id,
        "occluder_name": occluder_name,
        "oracle_frame_idx": oracle_frame_idx,
        "conditions": {
            "clean": {
                "scene_dir": _format_path_for_json(clean_dir, path_anchor),
                "scene_type": "clean",
                "image_retrieval": _build_image_retrieval(
                    clean_dir, camera_data_clean, oracle_frame_idx, path_anchor=path_anchor
                ),
                "questions": payload_clean["questions"],
            },
            _occ_condition_key(occlusion_scene_type): {
                "scene_dir": _format_path_for_json(occlusion_dir, path_anchor),
                "scene_type": occlusion_scene_type,
                "image_retrieval": _build_image_retrieval(
                    occlusion_dir, camera_data_occ, oracle_frame_idx, oracle_meta, path_anchor=path_anchor
                ),
                "questions": payload_occ["questions"],
            },
        },
    }


def build_aggregate_benchmark(
    batch_root: Path,
    units: list[dict[str, Any]],
    path_anchor: Path | None = None,
    benchmark_json: Path | None = None,
) -> dict[str, Any]:
    """
    One JSON for all occlusion runs under batch_root, grouped by scene_id then target_id.

    Structure:
      scenes[].scene_id
      scenes[].clean_scene_dir
      scenes[].targets[]            — one entry per unique (scene_id, target_id)
        .target_id / .target_name
        .referent_id / .referent_name
        .clean                      — evaluated ONCE per target (not per occluder)
        .occlusions[]               — one entry per occluder
            .unit_id
            .occluder_id / .occluder_name
            .occlusion_type         — "partial_occlusion" | "full_occlusion"
            .oracle_frame_idx
            .occlusion_dir
            .scene_dir / .image_retrieval / .questions
    """
    scene_map: dict[str, dict[str, Any]] = {}
    # target_map: (scene_id, target_id) -> target entry index in scene_map[sid]["targets"]
    target_map: dict[tuple[str, str], int] = {}

    for unit in units:
        sid = str(unit.get("scene_name", ""))
        if sid not in scene_map:
            clean_block = unit.get("conditions", {}).get("clean") or {}
            scene_map[sid] = {
                "scene_id": sid,
                "clean_scene_dir": clean_block.get("scene_dir", ""),
                "targets": [],
            }

        conditions = unit.get("conditions", {})
        occ_key = "partial_occlusion" if "partial_occlusion" in conditions else "full_occlusion"
        occ_block = conditions.get(occ_key) or {}
        occ_dir_s = str(occ_block.get("scene_dir", ""))
        occ_path = Path(occ_dir_s) if occ_dir_s else Path()

        target_id = unit.get("target_id", "")
        tkey = (sid, target_id)

        occ_entry = {
            "unit_id":             unit.get("unit_id"),
            "occluder_id":         unit.get("occluder_id"),
            "occluder_name":       unit.get("occluder_name"),
            "occlusion_type":      occ_key,
            "oracle_frame_idx":    unit.get("oracle_frame_idx"),
            "occlusion_dir":       occ_dir_s,
            "occlusion_folder_name": occ_path.name if occ_dir_s else "",
            "scene_dir":           occ_block.get("scene_dir"),
            "image_retrieval":     occ_block.get("image_retrieval"),
            "questions":           occ_block.get("questions"),
        }

        if tkey not in target_map:
            # First time we see this (scene, target): create entry with clean block
            clean_block = conditions.get("clean") or {}
            idx = len(scene_map[sid]["targets"])
            target_map[tkey] = idx
            # Extract referent from clean block meta (stored in payload meta)
            clean_meta = unit.get("conditions", {}).get("clean", {})
            scene_map[sid]["targets"].append({
                "target_id":    target_id,
                "target_name":  unit.get("target_name"),
                "clean":        clean_block,
                "occlusions":   [occ_entry],
            })
        else:
            # Same target, different occluder: append to occlusions only
            idx = target_map[tkey]
            scene_map[sid]["targets"][idx]["occlusions"].append(occ_entry)

    scenes = sorted(scene_map.values(), key=lambda s: s["scene_id"])
    for s in scenes:
        s["targets"].sort(key=lambda t: t.get("target_id") or "")
        for t in s["targets"]:
            t["occlusions"].sort(key=lambda o: o.get("occlusion_folder_name") or "")

    num_unique_targets = sum(len(s["targets"]) for s in scenes)
    num_occlusions = sum(len(t["occlusions"]) for s in scenes for t in s["targets"])
    meta_out: dict[str, Any] = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "num_scenes": len(scenes),
        "num_unique_targets": num_unique_targets,
        "num_occlusions": num_occlusions,
    }
    if path_anchor is not None:
        # Store anchor relative to this JSON's parent so the file has no absolute paths.
        if benchmark_json is not None:
            meta_out["path_anchor"] = (
                os.path.relpath(path_anchor.resolve(), benchmark_json.parent.resolve())
                .replace("\\", "/")
            )
            meta_out["path_anchor_relative_to"] = "benchmark_json_parent"
        else:
            meta_out["path_anchor"] = str(path_anchor.resolve())
        meta_out["batch_root"] = _format_path_for_json(batch_root, path_anchor) or "."
    else:
        meta_out["path_anchor"] = None
        meta_out["path_anchor_relative_to"] = None
        meta_out["batch_root"] = str(batch_root.resolve())

    return {
        "meta": meta_out,
        "scenes": scenes,
    }


# ──────────────────────────────────────────────
# Batch
# ──────────────────────────────────────────────

def _iter_occlusion_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        names = set(filenames)
        p = Path(dirpath)
        if ({"object_attributes.json", "camera_poses.json", "occlusion_meta.json"}.issubset(names)
                and p.name.startswith("occlusion_")):
            out.append(p)
    return sorted(out)


def _resolve_clean_scene_dir(occlusion_dir: Path, clean_root_override: Path | None = None) -> Path | None:
    """
    Folder with unoccluded layout: object_attributes.json + camera_poses.json.

    Supported layouts:
      1) Legacy: same parent as occlusion_dir/   (parent/object_attributes.json)
      2) Common export: .../occlusion_scenes/<scene_id>/occlusion_*  paired with
         .../clean_scenes/<scene_id>/  (sibling dataset root)
      3) Optional: --clean-root <DIR>  ->  DIR/<scene_id>/  where scene_id is occlusion_dir.parent.name
    """
    parent = occlusion_dir.parent
    oa, cp = parent / "object_attributes.json", parent / "camera_poses.json"
    if oa.exists() and cp.exists():
        return parent

    scene_id = parent.name
    occ_resolved = occlusion_dir.resolve()
    parts = occ_resolved.parts
    try:
        idx = next(i for i, p in enumerate(parts) if p == "occlusion_scenes")
    except StopIteration:
        idx = -1
    if idx >= 0:
        base = Path(*parts[:idx])
        # Try common sibling roots for clean scenes.
        for clean_root_name in ("clean_scenes", "clean_scene_layout"):
            cand = base / clean_root_name / scene_id
            oa2, cp2 = cand / "object_attributes.json", cand / "camera_poses.json"
            if oa2.exists() and cp2.exists():
                return cand

    if clean_root_override is not None:
        cand = clean_root_override.expanduser().resolve() / scene_id
        oa3, cp3 = cand / "object_attributes.json", cand / "camera_poses.json"
        if oa3.exists() and cp3.exists():
            return cand

    return None


def _annotation_key(occlusion_dir: Path) -> str:
    """Build the key used in annotation_results.json: '<scene_id>/<occ_folder>'."""
    return f"{occlusion_dir.parent.name}/{occlusion_dir.name}"


def _process_pair(
    occlusion_dir: Path,
    output_name_clean: str,
    output_name_occ: str,
    output_name_unit: str = "eval_unit.json",
    clean_root_override: Path | None = None,
    write_per_occlusion: bool = True,
    path_anchor: Path | None = None,
    occlusion_scene_type: str = "full_occlusion",
) -> dict[str, Any] | None:
    """Process one occlusion scene. Returns eval_unit dict if clean scene exists, else None."""
    objects_occ, camera_data_occ, meta_occ = _load_scene_assets(occlusion_dir)
    object_map_occ = {str(o.get("id") or o.get("thor_objectId") or ""): o for o in objects_occ}
    target_id, occluder_id = _resolve_target_and_occluder(occlusion_dir, object_map_occ, meta_occ)

    oracle_idx: int | None = None
    if meta_occ:
        oracle = meta_occ.get("oracle_frame") or {}
        if isinstance(oracle, dict) and "frame_idx" in oracle:
            oracle_idx = int(oracle["frame_idx"])

    # Occlusion-condition questions (scene_type reflects actual annotation label)
    payload_occ = generate_questions(occlusion_dir, scene_type=occlusion_scene_type, path_anchor=path_anchor)
    if write_per_occlusion:
        save_json(occlusion_dir / output_name_occ, payload_occ)
        print(f"[OK] {occlusion_dir.name} → {output_name_occ}")
    else:
        print(f"[built] {occlusion_dir.name} (full_occlusion)")

    # Clean scene questions — unoccluded layout (parent, clean_scenes/<id>, or --clean-root)
    clean_dir = _resolve_clean_scene_dir(occlusion_dir, clean_root_override=clean_root_override)
    if clean_dir is None:
        print(
            f"[SKIP] No clean scene (object_attributes.json + camera_poses.json) for {occlusion_dir.name}: "
            f"tried parent {occlusion_dir.parent!s}, parallel clean_scenes/{occlusion_dir.parent.name!s}, "
            f"and --clean-root if provided."
        )
        return None
    print(f"  clean_dir = {clean_dir}")

    _, camera_data_clean, _ = _load_scene_assets(clean_dir)
    payload_clean = generate_questions(
        clean_dir,
        scene_type="clean",
        target_id_override=target_id,
        occluder_id_override=occluder_id,
        oracle_frame_idx=oracle_idx,
        path_anchor=path_anchor,
    )
    if write_per_occlusion:
        save_json(occlusion_dir / output_name_clean, payload_clean)
        print(f"[OK] {occlusion_dir.name} → {output_name_clean}")

    # Eval unit — combines both conditions with image retrieval paths
    unit = build_eval_unit(
        occlusion_dir=occlusion_dir,
        clean_dir=clean_dir,
        payload_occ=payload_occ,
        payload_clean=payload_clean,
        target_id=target_id,
        occluder_id=occluder_id,
        oracle_frame_idx=oracle_idx,
        camera_data_occ=camera_data_occ,
        camera_data_clean=camera_data_clean,
        meta_occ=meta_occ,
        path_anchor=path_anchor,
        occlusion_scene_type=occlusion_scene_type,
    )
    if write_per_occlusion:
        save_json(occlusion_dir / output_name_unit, unit)
        print(f"[OK] {occlusion_dir.name} → {output_name_unit}")
    return unit


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate occlusion benchmark questions.")
    parser.add_argument("--input", default="", help="Single scene folder.")
    parser.add_argument(
        "--scene-type", default="full_occlusion",
        choices=["clean", "full_occlusion"],
        help="Scene type for single-scene mode.",
    )
    parser.add_argument("--target-id", default="", help="Explicit target object id.")
    parser.add_argument("--occluder-id", default="", help="Explicit occluder object id.")
    parser.add_argument("--oracle-frame-idx", type=int, default=None, help="Frame index to use.")
    parser.add_argument("--output", default="", help="Output JSON path (single mode).")
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Single mode: directory for questions_{scene_type}.json "
            "(default: occlusion_* folder when applicable, else --input folder)."
        ),
    )
    parser.add_argument("--batch-root", default="", help="Process all occlusion scenes under this root.")
    parser.add_argument(
        "--clean-root",
        default="",
        help=(
            "Batch mode: if clean assets are not under occlusion parent or clean_scenes/<scene_id>, "
            "set this to a directory containing <scene_id>/object_attributes.json "
            "(scene_id = parent folder name of each occlusion_*)."
        ),
    )
    parser.add_argument("--output-name-clean", default="questions_clean.json")
    parser.add_argument("--output-name-occ", default="questions_full_occlusion.json")
    parser.add_argument(
        "--aggregate-json",
        default="",
        help=(
            "Batch mode: write one combined JSON (grouped by scene_id → targets with clean + full_occlusion). "
            "By default this disables per-occlusion question files; use --also-write-per-occlusion to keep those."
        ),
    )
    parser.add_argument(
        "--also-write-per-occlusion",
        action="store_true",
        help="Batch mode: when using --aggregate-json, also write questions_*.json and eval_unit.json under each occlusion folder.",
    )
    parser.add_argument(
        "--annotation-results",
        default="",
        help=(
            "Path to annotation_results.json (from annotate_occlusion.py). "
            "Maps '<scene_id>/<occ_folder>' → label. "
            "Labels: no_occlude, partial_occlude, fully_occlude (→ full_occlusion), invalid_example (skipped). "
            "Scenes not present in the file are treated as full_occlusion."
        ),
    )
    parser.add_argument(
        "--path-root",
        default="",
        help=(
            "If set (absolute dir), scene_dir / image paths / batch_root are relative to this directory. "
            "In --aggregate-json output, meta.path_anchor is also relative to the benchmark JSON's parent "
            "(see meta.path_anchor_relative_to); resolve: Path(benchmark_json).parent / meta['path_anchor']."
        ),
    )
    args = parser.parse_args()

    if args.batch_root:
        root = Path(args.batch_root).expanduser().resolve()
        path_anchor = Path(args.path_root).expanduser().resolve() if args.path_root else None
        dirs = _iter_occlusion_dirs(root)
        if not dirs:
            raise FileNotFoundError(f"No occlusion scene folders found under: {root}")
        clean_ov = Path(args.clean_root).expanduser().resolve() if args.clean_root else None
        aggregate_path = Path(args.aggregate_json).expanduser().resolve() if args.aggregate_json else None
        write_per_occlusion = aggregate_path is None or args.also_write_per_occlusion

        # Load annotation labels if provided
        annotation_labels: dict[str, str] = {}
        if args.annotation_results:
            ann_path = Path(args.annotation_results).expanduser().resolve()
            annotation_labels = load_json(ann_path)
            print(f"[INFO] Loaded {len(annotation_labels)} annotation labels from {ann_path}")

        # Map annotation label → scene_type
        _LABEL_TO_SCENE_TYPE = {
            "no_occlude":      "no_occlude",
            "partial_occlude": "partial_occlude",
            "fully_occlude":   "full_occlusion",
        }

        units: list[dict[str, Any]] = []
        skipped = 0
        for d in dirs:
            key = _annotation_key(d)
            label = annotation_labels.get(key)
            if label in ("invalid_example", "no_occlude"):
                print(f"[SKIP] {label}: {key}")
                skipped += 1
                continue
            occ_scene_type = _LABEL_TO_SCENE_TYPE.get(label or "", "full_occlusion")
            u = _process_pair(
                d,
                args.output_name_clean,
                args.output_name_occ,
                clean_root_override=clean_ov,
                write_per_occlusion=write_per_occlusion,
                path_anchor=path_anchor,
                occlusion_scene_type=occ_scene_type,
            )
            if u is not None:
                units.append(u)
        if skipped:
            print(f"[INFO] Skipped {skipped} scenes (invalid_example or no_occlude)")
        if aggregate_path is not None:
            agg = build_aggregate_benchmark(
                root,
                units,
                path_anchor=path_anchor,
                benchmark_json=aggregate_path,
            )
            save_json(aggregate_path, agg)
            m = agg["meta"]
            print(f"[OK] Aggregate benchmark → {aggregate_path} ({m['num_unique_targets']} unique targets, {m['num_occlusions']} occlusions)")
        return

    if not args.input:
        raise ValueError("Provide --input or --batch-root.")

    path_anchor = Path(args.path_root).expanduser().resolve() if args.path_root else None
    scene_dir = _parse_scene_folder(Path(args.input).expanduser().resolve())
    payload = generate_questions(
        scene_dir,
        scene_type=args.scene_type,
        target_id_override=args.target_id or None,
        occluder_id_override=args.occluder_id or None,
        oracle_frame_idx=args.oracle_frame_idx,
        path_anchor=path_anchor,
    )
    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    elif args.output_dir:
        out_path = Path(args.output_dir).expanduser().resolve() / f"questions_{args.scene_type}.json"
    else:
        out_base = _default_single_output_dir(
            scene_dir,
            args.scene_type,
            args.target_id or None,
            args.occluder_id or None,
        )
        out_path = out_base / f"questions_{args.scene_type}.json"
    save_json(out_path, payload)
    print(f"[OK] Saved → {out_path}")


if __name__ == "__main__":
    main()
