"""
question_gen_distortion.py

Generate multiple-choice questions for distortion benchmark scenes.

Supports two distortion types:
  1. size_distance  — same-size objects appear different due to camera proximity
  2. foreshortening — 3D shapes appear distorted (circular→ellipse, tall→short, etc.)

Output structure (benchmark.json):
  {
    "meta": {...},
    "scenes": [
      {
        "scene_id": "...",
        "targets": [
          {
            # size_distance variant:
            "target_id": "coat_rack",
            "distortion_type": "size_distance",
            "clean_views":     [{view_entry}, ...],   # equidistant camera frames
            "distorted_views": [{view_entry}, ...],   # near-object frames

            # foreshortening variant:
            "target_id": "dining_table",
            "distortion_type": "foreshortening_rectangular",
            "reference":       {view_entry},           # overhead / unambiguous view
            "distorted_views": [{view_entry}, ...],   # low-angle distorted views
          }
        ]
      }
    ]
  }

Each view_entry:
  {
    "view_idx":        int,
    "view_type":       str,
    "scene_type":      "clean" | "distorted",
    "description":     str,
    "scene_dir":       str,
    "image_retrieval": {"sequence": str, "oracle_frame": {"frame_idx": int, "image_path": str}},
    "questions":       [question_dict],
  }
"""
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


CANNOT_DETERMINE      = "Cannot determine from this viewpoint"
CANNOT_DETERMINE_SIZE = "Cannot determine from this image"

# ── Question templates ─────────────────────────────────────────────────────────

_DISTORTION_QUESTION: dict[str, dict[str, Any]] = {
    "size_distance": {
        "question_template": "Which {name} is physically larger?",
        "options": [
            "The one closer to the camera is larger",
            "They are approximately the same physical size",
            "The one farther from the camera is larger",
            "Cannot determine",
        ],
        # GT is identical for every frame — physical size doesn't change with camera position
        "answer": "They are approximately the same physical size",
    },
    "foreshortening_circular": {
        "question_template": "What is the true top-down shape of this {name}?",
        "options": [
            "Circular (round)",
            "Elliptical (oval)",
            "Rectangular or square",
            CANNOT_DETERMINE,
        ],
        "reference_answer": "Circular (round)",
        "distorted_answer": CANNOT_DETERMINE,
    },
    "foreshortening_rectangular": {
        "question_template": "What is the true footprint shape of this {name}?",
        "options": [
            "Clearly elongated (much longer than wide)",
            "Roughly square (similar length and width)",
            "Clearly compressed (much wider than tall in this view)",
            CANNOT_DETERMINE,
        ],
        "reference_answer": "Clearly elongated (much longer than wide)",
        "distorted_answer": CANNOT_DETERMINE,
    },
    "foreshortening_tall": {
        "question_template": "Compared to its width, how tall is this {name} in reality?",
        "options": [
            "Much taller than wide",
            "Roughly the same height as width",
            "Much wider than tall",
            CANNOT_DETERMINE,
        ],
        "reference_answer": "Much taller than wide",
        "distorted_answer": CANNOT_DETERMINE,
    },
    "foreshortening_short": {
        "question_template": "Compared to its width, how tall is this {name} in reality?",
        "options": [
            "Much shorter than wide",
            "Roughly the same height as width",
            "Much taller than wide",
            CANNOT_DETERMINE,
        ],
        "reference_answer": "Much shorter than wide",
        "distorted_answer": CANNOT_DETERMINE,
    },
}


# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _format_path(path: Path | str | None, anchor: Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path).expanduser().resolve()
    if anchor is None:
        return str(p)
    return os.path.relpath(p, anchor.expanduser().resolve()).replace("\\", "/")


# ── Name helpers ───────────────────────────────────────────────────────────────

def _human_name(raw: str) -> str:
    """Generic type name: 'bar_stool-1 (bar)' → 'bar stool'"""
    s = re.sub(r"\s*\([^)]*\)\s*$", "", str(raw)).strip()
    s = re.sub(r"[_-]+", " ", s).strip()
    s = re.sub(r"\b\d+\b$", "", s).strip()
    s = re.sub(r"-\d+$", "", s).strip()
    return re.sub(r"\s+", " ", s).strip()


def _instance_name(raw: str) -> str:
    """Type name only: 'bar_stool-1 (bar)' → 'bar stool'"""
    s = re.sub(r"\s*\([^)]*\)\s*$", "", str(raw)).strip()  # strip room
    s = re.sub(r"-\d+$", "", s).strip()                     # strip trailing -N
    s = re.sub(r"[_]+", " ", s).strip()                     # underscores → spaces
    return re.sub(r"\s+", " ", s).strip()


def _short_name(raw: str) -> str:
    """Last word only: 'bar_stool-1 (bar)' → 'stool', 'coat_rack-2 (locker room)' → 'rack'"""
    full = _instance_name(raw)
    return full.split()[-1] if full else full


# ── MC shuffle ─────────────────────────────────────────────────────────────────

def _shuffle_options(
    options: list[str],
    answer_text: str,
    seed: str,
) -> tuple[list[str], str]:
    shuffled = list(options)
    random.Random(seed).shuffle(shuffled)
    letters = ["A", "B", "C", "D", "E", "F"]
    labeled = [f"{letters[i]}) {opt}" for i, opt in enumerate(shuffled)]
    gt_letter = letters[shuffled.index(answer_text)]
    return labeled, gt_letter


# ── Image path resolution ──────────────────────────────────────────────────────

def _find_view_image(variant_dir: Path, sequence: str, view_idx: int) -> Path | None:
    """Locate the rendered image for a specific view index."""
    candidates: list[Path] = []
    if sequence:
        seq_dir = variant_dir / sequence
        candidates += [
            seq_dir / f"view_{view_idx:03d}.png",
            seq_dir / f"view_{view_idx:03d}.jpg",
            seq_dir / f"{view_idx:03d}.png",
        ]
    candidates += [
        variant_dir / f"view_{view_idx:03d}.png",
        variant_dir / f"view_{view_idx:03d}.jpg",
        variant_dir / f"{view_idx:03d}.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _image_retrieval(
    variant_dir: Path,
    sequence: str,
    view_idx: int,
    anchor: Path | None,
) -> dict[str, Any]:
    img_path = _find_view_image(variant_dir, sequence, view_idx)
    return {
        "sequence": sequence,
        "oracle_frame": {
            "frame_idx":  view_idx,
            "image_path": _format_path(img_path, anchor) if img_path else None,
        },
    }


# ── Shape question builder (wall pairs) ──────────────────────────────────────

def _build_shape_question(
    view_idx:         int,
    view_type:        str,
    scene_type:       str,        # "clean" | "distorted"
    obj_type:         str,        # e.g. "wall art"
    variant_dir_name: str,
    target_id:        str,
    ar_a:             float | None = None,
    ar_b:             float | None = None,
    pair_category:    str = "wall",   # "wall" | "floor"
) -> dict[str, Any]:
    """
    Test for visual epistemic calibration on shape/proportions.

    clean     → symmetric camera; both appear same shape
                → GT: "Yes, same proportions"  (can determine)
    distorted → one is foreshortened by camera angle
                → GT: "Cannot determine from this image"  (should not conclude)

    Wall  — oblique camera foreshortens one picture's aspect ratio.
    Floor — perpendicular orientations make one object look narrower/wider.
    """
    question_id = f"{variant_dir_name}|{target_id}|view{view_idx:03d}|shape_aspect"
    seed        = f"{variant_dir_name}|{target_id}|{view_idx}|shape_aspect"

    if pair_category == "floor":
        question_text = (
            f"There are two {obj_type}s visible in this image. "
            f"Based on this image, can you determine whether they have the same "
            f"physical shape (proportions of length vs. width)?"
        )
        options = [
            f"Yes, they have the same physical proportions",
            f"No, one is physically longer/narrower than the other",
            f"No, one is physically wider/squarer than the other",
            "Cannot determine from this image",
        ]
    else:  # wall
        question_text = (
            f"There are two {obj_type}s visible in this image. "
            f"Based on this image, can you determine whether they have the same "
            f"physical shape (width-to-height proportions)?"
        )
        options = [
            f"Yes, they have the same physical proportions",
            f"No, one is physically more landscape (wider) than the other",
            f"No, one is physically more portrait (taller) than the other",
            "Cannot determine from this image",
        ]

    answer_text = (
        options[0]
        if scene_type == "clean"
        else "Cannot determine from this image"
    )

    labeled, gt_letter = _shuffle_options(options, answer_text, seed)

    return {
        "question_id":     question_id,
        "question":        question_text,
        "question_type":   "distortion_shape",
        "distortion_type": "shape_aspect",
        "view_type":       view_type,
        "scene_type":      scene_type,
        "pair_category":   pair_category,
        "ar_a":            ar_a,
        "ar_b":            ar_b,
        "answer_text":     answer_text,
        "answer_options":  labeled,
        "gt_answer":       gt_letter,
    }


# ── Wall shape question builder ───────────────────────────────────────────────

def _build_wall_shape_question(
    view_idx:         int,
    view_type:        str,
    scene_type:       str,
    obj_type:         str,
    variant_dir_name: str,
    target_id:        str,
) -> dict[str, Any]:
    """
    Wall pair shape / proportion question.

    Only generated for variants where annotation gt_larger == "same"
    (i.e. both pictures have the same longest side in the clean frame).
    Since the two pictures are the same size, they share similar proportions.

    clean     → GT: "Yes, they share similar proportions" (always, since only "same" variants reach here)
    distorted → GT: "Cannot determine from this image"
    (oblique camera angle foreshortens one picture, distorting apparent proportions)
    """
    qid  = f"{variant_dir_name}|{target_id}|view{view_idx:03d}|wall_shape"
    seed = f"{variant_dir_name}|{target_id}|{view_idx}|wall_shape"

    question_text = (
        f"There are two {obj_type}s visible in this image. "
        f"Based on this image, do they share similar proportions "
        f"(width-to-height ratio)?"
    )
    options = [
        "Yes, they share similar proportions",
        "No, one is noticeably wider relative to its height",
        "No, one is noticeably taller relative to its width",
        "Cannot determine from this image",
    ]

    if scene_type == "clean":
        answer_text = "Yes, they share similar proportions"
    else:
        answer_text = "Cannot determine from this image"

    labeled, gt_letter = _shuffle_options(options, answer_text, seed)
    return {
        "question_id":    qid,
        "question":       question_text,
        "question_type":  "wall_shape",
        "view_type":      view_type,
        "scene_type":     scene_type,
        "answer_text":    answer_text,
        "answer_options": labeled,
        "gt_answer":      gt_letter,
    }


# ── Referent helpers ─────────────────────────────────────────────────────────

# Preferred referent categories: large, identifiable, floor-standing objects
_REFERENT_KEYWORDS = [
    "table", "counter", "sofa", "desk", "bench", "cabinet",
    "shelf", "bookshelf", "rack", "wardrobe", "bed", "chair",
    "display", "island", "workbench", "dresser", "sideboard",
]

_SKIP_TYPES = {"wall", "floor", "ceiling", "window", "door",
               "light", "lamp", "vent", "outlet", "switch"}

MAX_FLOOR_Y = 1.2    # objects with centre y > this are wall-mounted


def _xz_dist(p1: dict, p2: dict) -> float:
    dx = float(p1["x"]) - float(p2["x"])
    dz = float(p1["z"]) - float(p2["z"])
    return math.sqrt(dx * dx + dz * dz)


def _base_type_qg(obj: dict) -> str:
    name = obj.get("object_name") or ""
    if name:
        return re.sub(r"[-_]\d+$", "", str(name)).strip().lower()
    raw = obj.get("id") or ""
    return re.split(r"[|_\-]", str(raw))[0].strip().lower()


def _human_obj_name(obj: dict) -> str:
    t = _base_type_qg(obj)
    return re.sub(r"[_]+", " ", t).strip()


def _find_referent(
    scene: dict,
    obj_a: dict,
    obj_b: dict,
    near_object: str | None = None,   # "a" | "b" — which target is close to camera
    cam_pos: dict | None = None,       # {"x": ..., "z": ...} of clean frame camera
    cam_yaw: float | None = None,      # yaw degrees of clean frame camera
    fov_deg: float = 100.0,
    min_dist_diff: float = 0.6,
    max_ref_dist:  float = 8.0,
) -> tuple[dict, str, str] | None:
    """
    Find a suitable third object to use as a spatial referent.

    Selection priority:
      1. Visible from the clean frame camera (within FOV)
      2. On the FAR side — opposite from near_object (camera moved away from it)
      3. Known large object type
      4. Largest distance difference to obj_a vs obj_b

    Returns (referent_obj, referent_name, closer_obj) where closer_obj = "a" | "b".
    Returns None if no good referent exists.
    """
    target_ids = {obj_a["id"], obj_b["id"]}
    pos_a = obj_a["position"]
    pos_b = obj_b["position"]
    mid_x = (float(pos_a["x"]) + float(pos_b["x"])) / 2
    mid_z = (float(pos_a["z"]) + float(pos_b["z"])) / 2
    mid   = {"x": mid_x, "z": mid_z}

    # Far object: opposite of near_object
    far_pos: dict | None = None
    if near_object == "a":
        far_pos = pos_b
    elif near_object == "b":
        far_pos = pos_a

    candidates = []
    for obj in scene.get("objects", []):
        oid = obj.get("id", "")
        if oid in target_ids:
            continue
        pos = obj.get("position", {})
        y   = float(pos.get("y", 99))
        if y > MAX_FLOOR_Y:
            continue
        t = _base_type_qg(obj)
        if t in _SKIP_TYPES:
            continue

        dist_to_mid = _xz_dist(pos, mid)
        if dist_to_mid > max_ref_dist:
            continue

        d_a = _xz_dist(pos, pos_a)
        d_b = _xz_dist(pos, pos_b)
        diff = abs(d_a - d_b)
        if diff < min_dist_diff:
            continue

        # ── FOV visibility check from clean frame camera ──────────────────
        outside_fov = 0   # 0 = visible (good), 1 = outside FOV
        if cam_pos is not None and cam_yaw is not None:
            to_x = float(pos.get("x", 0)) - float(cam_pos.get("x", 0))
            to_z = float(pos.get("z", 0)) - float(cam_pos.get("z", 0))
            dist_cam = math.sqrt(to_x ** 2 + to_z ** 2)
            if dist_cam > 0.1:
                fwd_x = math.sin(math.radians(float(cam_yaw)))
                fwd_z = math.cos(math.radians(float(cam_yaw)))
                cos_a = (fwd_x * to_x + fwd_z * to_z) / dist_cam
                if cos_a < math.cos(math.radians(fov_deg / 2)):
                    outside_fov = 1

        # ── Far-side preference ───────────────────────────────────────────
        not_far_side = 0   # 0 = on far side (good), 1 = on near side
        if far_pos is not None:
            far_x   = float(far_pos.get("x", mid_x))
            ref_x   = float(pos.get("x", mid_x))
            far_right = far_x > mid_x
            ref_right = ref_x > mid_x
            not_far_side = 0 if (far_right == ref_right) else 1

        priority = next((i for i, k in enumerate(_REFERENT_KEYWORDS) if k in t), 99)
        closer_obj = "a" if d_a < d_b else "b"
        candidates.append((outside_fov, not_far_side, priority, diff, obj, closer_obj))

    if not candidates:
        return None

    # Sort: in-FOV first, far-side first, known type first, largest diff first
    candidates.sort(key=lambda x: (x[0], x[1], x[2], -x[3]))
    _, _, _, _, ref_obj, closer_obj = candidates[0]
    ref_name = _human_obj_name(ref_obj)
    return ref_obj, ref_name, closer_obj


# ── Relative-relation question ────────────────────────────────────────────────

def _build_relative_relation_question(
    view_idx:         int,
    view_type:        str,
    scene_type:       str,
    obj_type:         str,
    variant_dir_name: str,
    target_id:        str,
    referent_name:    str,          # e.g. "dining table"
    closer_obj:       str,          # "a" | "b" — which object is physically closer
    a_side:           str | None,   # "left" | "right"
    b_side:           str | None,
) -> dict[str, Any]:
    """
    Which of the two objects is physically closer to a named referent?
    GT is a stable physical fact — does not change with camera position.
    """
    qid  = f"{variant_dir_name}|{target_id}|view{view_idx:03d}|relative_distance"
    seed = f"{variant_dir_name}|{target_id}|{view_idx}|relative_distance"

    question_text = (
        f"In this scene there is also a {referent_name}. "
        f"Which of the two {obj_type}s is physically closer to the {referent_name}?"
    )
    options = [
        "The one on the left",
        "The one on the right",
        "Both are at roughly the same distance from it",
        "Cannot determine from this image",
    ]

    closer_side = (a_side if closer_obj == "a" else b_side) if (a_side and b_side) else None
    if closer_side:
        answer_text = f"The one on the {closer_side}"
    else:
        answer_text = "Cannot determine from this image"

    labeled, gt_letter = _shuffle_options(options, answer_text, seed)
    return {
        "question_id":     qid,
        "question":        question_text,
        "question_type":   "relative_distance",
        "view_type":       view_type,
        "scene_type":      scene_type,
        "referent":        referent_name,
        "closer_obj":      closer_obj,
        "answer_text":     answer_text,
        "answer_options":  labeled,
        "gt_answer":       gt_letter,
    }


# ── Visibility question ───────────────────────────────────────────────────────

def _build_visibility_question(
    view_idx: int, view_type: str, scene_type: str,
    obj_type: str, variant_dir_name: str, target_id: str,
) -> dict[str, Any]:
    """
    How many objects of this type are visible?
    GT is always "Two" — we only generate variants where both objects are in-frame.
    This is a control question; the answer should not change between clean and distorted.
    """
    qid  = f"{variant_dir_name}|{target_id}|view{view_idx:03d}|visibility"
    seed = f"{variant_dir_name}|{target_id}|{view_idx}|visibility"
    question_text = f"How many {obj_type}s can you see in this image?"
    options = ["One", "Two", "Three or more", "None visible"]
    answer_text = "Two"
    labeled, gt_letter = _shuffle_options(options, answer_text, seed)
    return {
        "question_id":    qid,
        "question":       question_text,
        "question_type":  "visibility",
        "view_type":      view_type,
        "scene_type":     scene_type,
        "answer_text":    answer_text,
        "answer_options": labeled,
        "gt_answer":      gt_letter,
    }


# ── Relative-position question ────────────────────────────────────────────────

def _build_position_question(
    view_idx: int, view_type: str, scene_type: str,
    obj_type: str, variant_dir_name: str, target_id: str,
    a_side: str | None, b_side: str | None,
) -> dict[str, Any]:
    """
    Which object is on which side?
    GT depends on a_side/b_side — should be stable across clean and distorted frames
    for the same variant (minor lateral shift does not flip left/right).
    """
    qid  = f"{variant_dir_name}|{target_id}|view{view_idx:03d}|position"
    seed = f"{variant_dir_name}|{target_id}|{view_idx}|position"
    question_text = (
        f"In this image there are two {obj_type}s. "
        f"Which of the following best describes their horizontal arrangement?"
    )
    options = [
        "One on the left, one on the right, at roughly the same height",
        "One is directly above the other",
        "They overlap or are very close together",
        "Cannot determine their relative positions",
    ]
    # Our generation always places objects side by side at similar heights
    answer_text = "One on the left, one on the right, at roughly the same height"
    labeled, gt_letter = _shuffle_options(options, answer_text, seed)
    return {
        "question_id":    qid,
        "question":       question_text,
        "question_type":  "relative_position",
        "view_type":      view_type,
        "scene_type":     scene_type,
        "a_side":         a_side,
        "b_side":         b_side,
        "answer_text":    answer_text,
        "answer_options": labeled,
        "gt_answer":      gt_letter,
    }


# ── Depth question ────────────────────────────────────────────────────────────

def _build_depth_question(
    view_idx: int, view_type: str, scene_type: str,
    obj_type: str, variant_dir_name: str, target_id: str,
    near_object: str | None,   # "a" | "b" | None
    a_side: str | None,        # "left" | "right"
    b_side: str | None,
) -> dict[str, Any]:
    """
    Which object appears closer to the camera?
    GT:
      clean     → equidistant camera → "roughly the same distance"
      distorted → camera shifted toward near_object → "the one on the [near_side]"
    This is the true/correct answer (not an illusion) — tests depth perception.
    """
    qid  = f"{variant_dir_name}|{target_id}|view{view_idx:03d}|depth"
    seed = f"{variant_dir_name}|{target_id}|{view_idx}|depth"
    question_text = (
        f"Which of the two {obj_type}s appears to be physically closer to the camera?"
    )
    options = [
        "The one on the left",
        "The one on the right",
        "They appear to be at roughly the same distance from the camera",
        "Cannot determine",
    ]
    if scene_type == "clean" or near_object is None:
        answer_text = "They appear to be at roughly the same distance from the camera"
    else:
        near_side = a_side if near_object == "a" else b_side
        answer_text = f"The one on the {near_side}" if near_side else \
                      "They appear to be at roughly the same distance from the camera"
    labeled, gt_letter = _shuffle_options(options, answer_text, seed)
    return {
        "question_id":    qid,
        "question":       question_text,
        "question_type":  "depth",
        "view_type":      view_type,
        "scene_type":     scene_type,
        "near_object":    near_object,
        "answer_text":    answer_text,
        "answer_options": labeled,
        "gt_answer":      gt_letter,
    }


# ── Size-distance question builder ────────────────────────────────────────────

def _fmt_type(raw: str) -> str:
    """'individual_chair' → 'individual chair', 'worktable' → 'worktable'"""
    return re.sub(r"[_]+", " ", str(raw)).strip()


def _build_size_distance_question(
    view_idx:           int,
    view_type:          str,
    scene_type:         str,        # "clean" | "distorted"
    type_larger:        str,        # human-readable type (same for both objects)
    type_smaller:       str,        # same as type_larger for same-type pairs
    appears_larger:     str,        # "a"/"b"/"equal" (for logging)
    variant_dir_name:   str,
    target_id:          str,
    a_side:             str | None,   # "left" | "right"
    b_side:             str | None,
    near_object:        str | None,   # "a" | "b" | None
    pair_size_relation: str = "same",  # "same" | "different"
    wall_larger_obj:    str | None = None,  # "a" | "b" | "same" | None
    wall_dimension:     str | None = None,  # "horizontal" | "vertical" | None (floor or same)
) -> dict[str, Any]:
    """
    Test for visual epistemic calibration.

    Floor pairs: same model — GT clean="same physical size", distorted="Cannot determine"
    Wall pairs (same):  GT clean="same longest side", distorted="Cannot determine"
    Wall pairs (differ, horizontal): GT clean="left/right is wider", distorted="Cannot determine"
    Wall pairs (differ, vertical):   GT clean="left/right is taller", distorted="Cannot determine"
    """
    question_id = f"{variant_dir_name}|{target_id}|view{view_idx:03d}|size_distance"
    seed        = f"{variant_dir_name}|{target_id}|{view_idx}|size_distance"

    obj = type_larger

    if wall_larger_obj in ("a", "b", "same"):
        if wall_larger_obj == "same":
            # Wall pair, annotated same → "longest side" question
            question_text = (
                f"There are two {obj}s in this image. "
                f"Based on this image, which one has a longer longest side?"
            )
            options = [
                f"They appear to have the same longest side",
                f"The one on the left has a longer longest side",
                f"The one on the right has a longer longest side",
                "Cannot determine from this image",
            ]
            if scene_type == "clean":
                answer_text = "They appear to have the same longest side"
            else:
                answer_text = "Cannot determine from this image"

        elif wall_dimension == "horizontal":
            # Wall pair, annotated left/right, horizontal dimension → "wider" question
            question_text = (
                f"There are two {obj}s in this image. "
                f"Based on this image, which one is wider?"
            )
            options = [
                "They appear to be the same width",
                "The one on the left is wider",
                "The one on the right is wider",
                "Cannot determine from this image",
            ]
            if scene_type == "clean":
                if wall_larger_obj == "a" and a_side:
                    answer_text = f"The one on the {a_side} is wider"
                elif wall_larger_obj == "b" and b_side:
                    answer_text = f"The one on the {b_side} is wider"
                else:
                    answer_text = "They appear to be the same width"
            else:
                answer_text = "Cannot determine from this image"

        else:
            # wall_dimension == "vertical" → "taller" question
            question_text = (
                f"There are two {obj}s in this image. "
                f"Based on this image, which one is taller?"
            )
            options = [
                "They appear to be the same height",
                "The one on the left is taller",
                "The one on the right is taller",
                "Cannot determine from this image",
            ]
            if scene_type == "clean":
                if wall_larger_obj == "a" and a_side:
                    answer_text = f"The one on the {a_side} is taller"
                elif wall_larger_obj == "b" and b_side:
                    answer_text = f"The one on the {b_side} is taller"
                else:
                    answer_text = "They appear to be the same height"
            else:
                answer_text = "Cannot determine from this image"

    else:
        # Floor pair: same model — ask about physical size
        question_text = (
            f"There are two {obj}s in this image. "
            f"Based on this image, can you determine their physical sizes?"
        )
        options = [
            f"They are the same physical size",
            f"The {obj} on the left is physically larger",
            f"The {obj} on the right is physically larger",
            "Cannot determine from this image",
        ]
        if scene_type == "clean":
            answer_text = "They are the same physical size"
        else:
            answer_text = "Cannot determine from this image"

    labeled, gt_letter = _shuffle_options(options, answer_text, seed)

    # Record which side the near (appearing-larger) object is on, for analysis
    near_side = (a_side if near_object == "a" else b_side) if near_object else None

    return {
        "question_id":        question_id,
        "question":           question_text,
        "question_type":      "distortion_size_distance",
        "distortion_type":    "size_distance",
        "view_type":          view_type,
        "scene_type":         scene_type,
        "pair_size_relation": pair_size_relation,
        "appears_larger":     appears_larger,
        "near_side":          near_side,
        "type_larger":        type_larger,
        "answer_text":        answer_text,
        "answer_options":     labeled,
        "gt_answer":          gt_letter,
    }


# ── Foreshortening question builder ───────────────────────────────────────────

def _build_question(
    distortion_type: str,
    target_name: str,
    view_idx: int,
    view_type: str,
    scene_type: str,
    variant_dir_name: str,
    target_id: str,
) -> dict[str, Any]:
    tmpl = _DISTORTION_QUESTION.get(distortion_type)
    if tmpl is None:
        raise ValueError(f"Unknown distortion_type: {distortion_type!r}")

    question_text = tmpl["question_template"].format(name=target_name)
    options: list[str] = [o.format(name=target_name) for o in tmpl["options"]]
    is_reference  = (scene_type == "clean")
    answer_text   = tmpl["reference_answer"] if is_reference else tmpl["distorted_answer"]

    seed = f"{variant_dir_name}|{target_id}|{view_idx}|{distortion_type}"
    labeled, gt_letter = _shuffle_options(options, answer_text, seed)
    question_id = f"{variant_dir_name}|{target_id}|view{view_idx:03d}|{distortion_type}"

    return {
        "question_id":     question_id,
        "question":        question_text,
        "question_type":   "distortion_foreshortening",
        "distortion_type": distortion_type,
        "view_type":       view_type,
        "scene_type":      scene_type,
        "answer_text":     answer_text,
        "answer_options":  labeled,
        "gt_answer":       gt_letter,
    }


# ── Size-distance variant ──────────────────────────────────────────────────────

def _pick_best_frames(frames: list[dict], one_distorted: bool) -> list[dict]:
    """
    Filter camera frames to keep at most 1 clean + 1 distorted frame.
    For distorted, pick deterministically using list order (near=b first, near=a second).
    When one_distorted=False, keep all frames.
    Missing scene_type defaults to "distorted" (matches generation-time convention).
    """
    if not one_distorted:
        return frames

    def _stype(f: dict) -> str:
        return str(f.get("scene_type") or "distorted")

    clean_frames     = [f for f in frames if _stype(f) == "clean"]
    distorted_frames = [f for f in frames if _stype(f) == "distorted"]
    best_clean       = clean_frames[:1]
    best_distorted   = distorted_frames[:1]  # take first distorted (near=b by default)
    return best_clean + best_distorted


def _generate_size_distance_variant(
    variant_dir: Path,
    frame_meta: dict[str, Any],
    anchor: Path | None,
    one_distorted: bool = False,
    wall_annotations: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    Parse a distortion_sizedistance_* folder (cross-type pairs).
    Reads frames from camera_poses.json (multiview sequence).
    Separates clean (equidistant) vs distorted (near-object) frames.

    GT per frame:
      clean / near_larger  → physically larger type appears larger (correct perception)
      near_smaller         → physically smaller type appears larger (size-distance illusion)
    """
    cam_path = variant_dir / "camera_poses.json"
    if not cam_path.exists():
        print(f"  [WARN] No camera_poses.json in {variant_dir.name}")
        return None

    try:
        cam_data = load_json(cam_path)
    except Exception as e:
        print(f"  [WARN] Cannot load camera_poses: {e}")
        return None

    # Both objects are the same type (same-type pair, no resize)
    obj_type   = _fmt_type(str(frame_meta.get("object_type", "")))
    type_larger  = obj_type
    type_smaller = obj_type

    # target_id: use object type
    target_id = obj_type

    distortion_sub_type = str(frame_meta.get("distortion_sub_type", "size"))
    pair_cat            = str(frame_meta.get("pair_category", "floor"))
    pair_size_relation  = str(frame_meta.get("pair_size_relation", "same"))
    ar_a = frame_meta.get("aspect_ratio_a")
    ar_b = frame_meta.get("aspect_ratio_b")

    # For wall pairs: use human annotation to determine size GT and which dimension to compare.
    # annotation format: {"dimension": "horizontal"|"vertical", "gt_larger": "left"|"right"|"same"}
    # gt_larger "left" → object A (always on left in clean frame) has longer side in that dimension
    # gt_larger "right" → object B
    # If no annotation for this variant, skip size question for wall pairs.
    wall_larger_obj: str | None = None   # "a" | "b" | "same" | None
    wall_dimension:  str | None = None   # "horizontal" | "vertical" | None

    if pair_cat == "wall":
        vkey = f"{variant_dir.parent.name}/{variant_dir.name}"
        wall_ann = (wall_annotations or {}).get(vkey)
        if wall_ann:
            gt_larger  = wall_ann.get("gt_larger")   # "left" | "right" | "same"
            ann_dim    = wall_ann.get("dimension")    # "horizontal" | "vertical"
            if gt_larger == "same":
                wall_larger_obj = "same"
                wall_dimension  = None   # doesn't matter for same
            elif gt_larger == "left":
                wall_larger_obj = "a"
                wall_dimension  = ann_dim
            elif gt_larger == "right":
                wall_larger_obj = "b"
                wall_dimension  = ann_dim

    # floor pairs always ask size; wall pairs only when annotation exists
    ask_size  = (pair_cat == "floor") or (pair_cat == "wall" and wall_larger_obj is not None)
    # wall shape (proportion) question: only when annotated "same" (both pictures same size → same proportions)
    ask_shape = (pair_cat == "wall" and wall_larger_obj == "same")

    # Build lookup from frame_meta.frames to get a_side / b_side per frame
    fm_frames: dict[int, dict] = {
        int(fr["frame_idx"]): fr
        for fr in (frame_meta.get("frames") or [])
        if isinstance(fr, dict) and "frame_idx" in fr
    }


    sequence = (
        cam_data.get("camera_frame_for_modality_eval")
        or cam_data.get("primary_sequence")
        or "multiview"
    )
    all_frames = (cam_data.get(sequence) or {}).get("frames") or []
    if not all_frames:
        print(f"  [WARN] No frames in camera_poses for {variant_dir.name}")
        return None

    # ── Referent: find once per variant ──────────────────────────────────────────
    referent_result: tuple | None = None
    obj_a_id = frame_meta.get("object_a_id", "")
    obj_b_id = frame_meta.get("object_b_id", "")
    scene_path = variant_dir / "scene.json"
    if pair_cat == "floor" and scene_path.exists() and obj_a_id and obj_b_id:
        try:
            scene_data = load_json(scene_path)
            obj_lookup = {o["id"]: o for o in scene_data.get("objects", [])}
            obj_a_dict = obj_lookup.get(obj_a_id)
            obj_b_dict = obj_lookup.get(obj_b_id)
            if obj_a_dict and obj_b_dict:
                # Clean frame camera pose for visibility check
                clean_frame_data = next(
                    (f for f in all_frames if str(f.get("scene_type", "")) == "clean"), None
                )
                cam_pos = clean_frame_data.get("position") if clean_frame_data else None
                cam_yaw = (clean_frame_data.get("rotation") or {}).get("y") if clean_frame_data else None
                # Dominant near_object from first distorted frame
                dist_frames = [f for f in all_frames if str(f.get("scene_type", "")) == "distorted"]
                near_obj_dominant = dist_frames[0].get("near_object") if dist_frames else None
                referent_result = _find_referent(
                    scene_data, obj_a_dict, obj_b_dict,
                    near_object=near_obj_dominant,
                    cam_pos=cam_pos,
                    cam_yaw=cam_yaw,
                )
        except Exception as e:
            print(f"  [WARN] Could not find referent for {variant_dir.name}: {e}")

    frames = _pick_best_frames(all_frames, one_distorted)

    clean_views:     list[dict[str, Any]] = []
    distorted_views: list[dict[str, Any]] = []

    for frame in frames:
        if not isinstance(frame, dict):
            continue
        view_idx    = int(frame.get("frame_idx", 0))
        view_type   = str(frame.get("view_type", f"view_{view_idx:03d}"))
        scene_type  = str(frame.get("scene_type", "distorted"))
        description = str(frame.get("description", ""))

        # near_object: "a", "b", or None (clean).  larger_appears: same values.
        near_object    = frame.get("near_object")     # "a" / "b" / None
        larger_appears = frame.get("larger_appears")  # "a" / "b" / None (clean = None)
        appears_larger = larger_appears or "equal"    # for logging only

        # Look up a_side / b_side from frame_meta (not stored in camera_poses)
        fm_fr  = fm_frames.get(view_idx, {})
        a_side = fm_fr.get("a_side")
        b_side = fm_fr.get("b_side")

        ir = _image_retrieval(variant_dir, sequence, view_idx, anchor)

        # Build question battery for this frame
        questions: list[dict[str, Any]] = []

        # 1. Visibility — always; GT never changes
        questions.append(_build_visibility_question(
            view_idx, view_type, scene_type,
            obj_type, variant_dir.name, target_id,
        ))

        # 2. Size — floor pairs always; wall pairs only when annotated
        if ask_size:
            questions.append(_build_size_distance_question(
                view_idx, view_type, scene_type,
                type_larger, type_smaller, appears_larger,
                variant_dir.name, target_id,
                a_side=a_side, b_side=b_side,
                near_object=near_object,
                pair_size_relation=pair_size_relation,
                wall_larger_obj=wall_larger_obj,
                wall_dimension=wall_dimension,
            ))

        # 3. Shape — wall pairs where annotation is "same" only
        if ask_shape:
            questions.append(_build_wall_shape_question(
                view_idx, view_type, scene_type,
                obj_type, variant_dir.name, target_id,
            ))

        # 4. Relative relation — floor pairs only, when a suitable referent exists
        if pair_cat == "floor" and referent_result is not None:
            ref_obj, ref_name, closer_obj = referent_result
            questions.append(_build_relative_relation_question(
                view_idx, view_type, scene_type,
                obj_type, variant_dir.name, target_id,
                ref_name, closer_obj,
                a_side, b_side,
            ))

        entry = {
            "view_idx":        view_idx,
            "view_type":       view_type,
            "scene_type":      scene_type,
            "near_object":     near_object,
            "larger_appears":  larger_appears,
            "appears_larger":  appears_larger,
            "description":     description,
            "scene_dir":       _format_path(variant_dir, anchor),
            "image_retrieval": ir,
            "questions":       questions,
        }

        if scene_type == "clean":
            clean_views.append(entry)
        else:
            distorted_views.append(entry)

    if not clean_views and not distorted_views:
        print(f"  [WARN] No valid frames in {variant_dir.name}")
        return None

    return {
        "target_id":           target_id,
        "target_name":         f"{type_larger} vs {type_smaller}",
        "distortion_type":     "size_distance",
        "distortion_sub_type": distortion_sub_type,  # "size" | "shape"
        "variant_dir":         _format_path(variant_dir, anchor),
        "pair_category":       frame_meta.get("pair_category", "floor"),
        "type_larger":         type_larger,
        "type_smaller":        type_smaller,
        "object_a_id":         frame_meta.get("object_a_id", ""),
        "object_b_id":         frame_meta.get("object_b_id", ""),
        "dims_a":              frame_meta.get("dims_a"),
        "dims_b":              frame_meta.get("dims_b"),
        "angle_diff":          frame_meta.get("angle_diff"),
        # shape-specific
        "aspect_ratio_a":      ar_a,
        "aspect_ratio_b":      ar_b,
        "referent_name":       referent_result[1] if referent_result else None,
        "referent_closer_obj": referent_result[2] if referent_result else None,
        "distortion_meta":     frame_meta.get("summary", ""),
        "clean_views":         clean_views,
        "distorted_views":     distorted_views,
    }


# ── Foreshortening variant ─────────────────────────────────────────────────────

def _generate_foreshortening_variant(
    variant_dir: Path,
    frame_meta: dict[str, Any],
    anchor: Path | None,
) -> dict[str, Any] | None:
    """
    Parse a distortion_foreshortening_* folder.
    view_idx 0 → reference (scene_type="clean"), rest → distorted.
    """
    cam_path = variant_dir / "camera_poses.json"
    if not cam_path.exists():
        print(f"  [WARN] No camera_poses.json in {variant_dir.name}")
        return None

    try:
        cam_data = load_json(cam_path)
    except Exception as e:
        print(f"  [WARN] Cannot load camera_poses: {e}")
        return None

    distortion_type = str(frame_meta.get("distortion_type", ""))
    target_id       = str(frame_meta.get("target_id", ""))
    target_name     = _human_name(target_id)

    sequence = (
        cam_data.get("camera_frame_for_modality_eval")
        or cam_data.get("primary_sequence")
        or "multiview"
    )
    frames = (cam_data.get(sequence) or {}).get("frames") or []
    if not frames:
        print(f"  [WARN] No frames in {cam_path}")
        return None

    # Build description lookup from frame_meta.json
    fm_lookup: dict[int, dict] = {}
    for fm in (frame_meta.get("frames") or []):
        if isinstance(fm, dict) and fm.get("frame_idx") is not None:
            fm_lookup[int(fm["frame_idx"])] = fm

    reference_entry: dict[str, Any] | None = None
    distorted_views: list[dict[str, Any]] = []

    for frame in frames:
        if not isinstance(frame, dict):
            continue
        view_idx   = int(frame.get("frame_idx", 0))
        view_type  = str(frame.get("view_type", f"view_{view_idx:03d}"))
        description = str(frame.get("description", ""))
        if not description:
            description = str(fm_lookup.get(view_idx, {}).get("description", ""))

        # For foreshortening: scene_type from frame, else infer from view_idx
        scene_type = str(frame.get("scene_type", "clean" if view_idx == 0 else "distorted"))

        ir = _image_retrieval(variant_dir, sequence, view_idx, anchor)
        q  = _build_question(
            distortion_type, target_name, view_idx, view_type,
            scene_type, variant_dir.name, target_id,
        )

        entry = {
            "view_idx":        view_idx,
            "view_type":       view_type,
            "scene_type":      scene_type,
            "description":     description,
            "scene_dir":       _format_path(variant_dir, anchor),
            "image_retrieval": ir,
            "questions":       [q],
        }

        if view_idx == 0:
            reference_entry = entry
        else:
            distorted_views.append(entry)

    if reference_entry is None:
        print(f"  [WARN] No view_000 (reference) in {variant_dir.name}")
        return None

    return {
        "target_id":        target_id,
        "target_name":      target_name,
        "distortion_type":  distortion_type,
        "variant_dir":      _format_path(variant_dir, anchor),
        "distortion_meta":  frame_meta.get("summary", ""),
        "reference":        reference_entry,
        "distorted_views":  distorted_views,
    }


# ── Dispatcher ─────────────────────────────────────────────────────────────────

def generate_distortion_variant(
    variant_dir: Path,
    anchor: Path | None = None,
    one_distorted: bool = False,
    wall_annotations: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    meta_path = variant_dir / "frame_meta.json"
    if not meta_path.exists():
        return None

    try:
        frame_meta = load_json(meta_path)
    except Exception as e:
        print(f"  [WARN] Cannot load frame_meta: {e}")
        return None

    distortion_type = str(frame_meta.get("distortion_type", ""))

    if not distortion_type:
        print(f"  [WARN] Missing distortion_type in {meta_path}")
        return None

    if distortion_type == "size_distance":
        return _generate_size_distance_variant(variant_dir, frame_meta, anchor,
                                               one_distorted=one_distorted,
                                               wall_annotations=wall_annotations)
    elif distortion_type in _DISTORTION_QUESTION:
        return _generate_foreshortening_variant(variant_dir, frame_meta, anchor)
    else:
        print(f"  [SKIP] Unsupported distortion_type {distortion_type!r} in {variant_dir.name}")
        return None


# ── Benchmark builder ──────────────────────────────────────────────────────────

def load_annotation_keep_set(annotation_path: Path) -> set[str] | None:
    """
    Load annotation results JSON and return the set of variant keys with status='keep'.
    Keys are "scene_id/variant_name" (forward-slash joined).
    Returns None if annotation_path is not given.
    """
    if not annotation_path or not annotation_path.is_file():
        return None
    data = load_json(annotation_path)
    if not isinstance(data, dict):
        return None
    return {k for k, v in data.items() if isinstance(v, dict) and v.get("status") == "keep"}


def build_distortion_benchmark(
    distortion_root: Path,
    anchor: Path | None = None,
    distortion_type_filter: str = "",
    one_distorted: bool = False,
    annotation_keep: set[str] | None = None,
    size_only: bool = False,
    wall_annotations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scenes: list[dict[str, Any]] = []
    scene_dirs = sorted(d for d in distortion_root.iterdir() if d.is_dir())

    total_targets   = 0
    total_questions = 0

    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        variant_dirs = sorted(
            d for d in scene_dir.iterdir()
            if d.is_dir() and d.name.startswith("distortion_")
        )
        if not variant_dirs:
            continue

        targets: list[dict[str, Any]] = []
        for vdir in variant_dirs:
            # Optional filter by distortion type substring
            if distortion_type_filter and distortion_type_filter not in vdir.name:
                continue

            # Optional annotation filter: only include keep variants
            if annotation_keep is not None:
                vkey = f"{scene_id}/{vdir.name}"
                if vkey not in annotation_keep:
                    continue

            entry = generate_distortion_variant(vdir, anchor, one_distorted=one_distorted,
                                                wall_annotations=wall_annotations)
            if entry is None:
                continue

            # Skip shape (90°) variants if size_only
            if size_only and entry.get("distortion_sub_type") == "shape":
                continue

            targets.append(entry)

            # Count questions
            dtype = entry.get("distortion_type", "")
            if dtype == "size_distance":
                for v in entry.get("clean_views", []) + entry.get("distorted_views", []):
                    total_questions += len(v.get("questions", []))
            else:
                for v in [entry.get("reference")] + entry.get("distorted_views", []):
                    if v:
                        total_questions += len(v.get("questions", []))

        if not targets:
            continue

        scenes.append({
            "scene_id":  scene_id,
            "scene_dir": _format_path(scene_dir, anchor),
            "targets":   targets,
        })
        total_targets += len(targets)
        print(f"  [{scene_id}] {len(targets)} variant(s)")

    meta: dict[str, Any] = {
        "benchmark_type":  "distortion",
        "generated_at":    datetime.datetime.now().isoformat(timespec="seconds"),
        "distortion_root": _format_path(distortion_root, anchor),
        "num_scenes":      len(scenes),
        "num_targets":     total_targets,
        "num_questions":   total_questions,
    }
    if anchor is not None:
        meta["path_anchor"] = str(anchor)

    return {"meta": meta, "scenes": scenes}


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate distortion benchmark questions."
    )
    parser.add_argument(
        "--distortion-root",
        default="/Users/zhangyue/Desktop/Holodeck/distortion_scenes_layout",
    )
    parser.add_argument(
        "--output",
        default="/Users/zhangyue/Desktop/Holodeck/rendered_scene/distortion_benchmark.json",
    )
    parser.add_argument(
        "--anchor", default="",
        help="Store paths relative to this dir. Defaults to output parent dir.",
    )
    parser.add_argument(
        "--distortion-type", default="",
        help="Filter variants by type substring, e.g. sizedistance or foreshortening",
    )
    parser.add_argument(
        "--one-distorted", action="store_true",
        help="Keep only 1 clean + 1 distorted frame per variant (reduces question count ~3x).",
    )
    parser.add_argument(
        "--size-only", action="store_true",
        help="Only include size variants (parallel-facing pairs); skip 90° shape variants.",
    )
    parser.add_argument(
        "--annotations",
        default="/Users/zhangyue/Desktop/Holodeck/rendered_scene/distortion_annotation_results.json",
        help="Annotation results JSON. Only variants with status='keep' are included. "
             "Pass empty string to disable filtering.",
    )
    parser.add_argument(
        "--wall-annotations",
        default="/Users/zhangyue/Desktop/Holodeck/rendered_scene/wall_size_annotation.json",
        help="Wall picture size annotation JSON (two-step: dimension + gt_larger). "
             "Used to set GT for wall pair size questions. Pass empty string to disable.",
    )
    args = parser.parse_args()

    distortion_root = Path(args.distortion_root).expanduser().resolve()
    output_path     = Path(args.output).expanduser().resolve()

    if not distortion_root.is_dir():
        raise FileNotFoundError(f"distortion-root not found: {distortion_root}")

    anchor: Path | None = (
        Path(args.anchor).expanduser().resolve()
        if args.anchor.strip()
        else output_path.parent.resolve()
    )

    print(f"Scanning: {distortion_root}")
    print(f"Anchor:   {anchor}")

    ann_path = Path(args.annotations).expanduser().resolve() if args.annotations.strip() else None
    annotation_keep = load_annotation_keep_set(ann_path) if ann_path else None
    if annotation_keep is not None:
        print(f"Annotations: {ann_path}  ({len(annotation_keep)} keep variants)")
    else:
        print("Annotations: disabled (all variants included)")

    wall_ann_path = Path(args.wall_annotations).expanduser().resolve() if args.wall_annotations.strip() else None
    wall_annotations: dict[str, Any] | None = None
    if wall_ann_path and wall_ann_path.is_file():
        try:
            wall_annotations = load_json(wall_ann_path)
            print(f"Wall annotations: {wall_ann_path}  ({len(wall_annotations)} entries)")
        except Exception as e:
            print(f"[WARN] Could not load wall annotations: {e}")
    else:
        print("Wall annotations: not found (wall size questions will be skipped)")

    benchmark = build_distortion_benchmark(distortion_root, anchor, args.distortion_type,
                                           one_distorted=args.one_distorted,
                                           annotation_keep=annotation_keep,
                                           size_only=args.size_only,
                                           wall_annotations=wall_annotations)
    save_json(output_path, benchmark)

    m = benchmark["meta"]
    print(f"\nSaved → {output_path}")
    print(f"  scenes={m['num_scenes']}  targets={m['num_targets']}  questions={m['num_questions']}")


if __name__ == "__main__":
    main()
