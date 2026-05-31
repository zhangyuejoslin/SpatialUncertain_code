"""
gen_distortion_sizedistance.py  (v2 — scale-based)

Size-Distance illusion via object scaling:
  1. Pick a floor-standing object from the scene
  2. Clone it with scale = SCALE_SMALL (0.5x) placed PLACE_DIST metres away
  3. Sample clean (equidistant) + distorted (near-small / near-large) cameras

Modified scene.json (copy only — original untouched) contains both:
  • the original object  → "large" (scale 1.0, physically bigger)
  • the cloned object    → "small" (scale 0.5, physically smaller)

Question GT:
  clean / near_large  → larger-looking one IS physically bigger  (vision correct)
  near_small          → smaller-looking one is ACTUALLY bigger   (size-distance illusion)
"""

import json
import math
import copy
import re
import itertools
import numpy as np
from pathlib import Path

# ── config ─────────────────────────────────────────────────────────────────────

SCALE_SMALL          = 0.05   # clone is this fraction of original (20x ratio)
PLACE_DIST           = 2.0    # metres between large and small objects
MIN_OBJ_SIZE         = 0.2    # minimum footprint (m) — lowered to catch wall pictures
MAX_FLOOR_OBJ_Y      = 1.2    # floor objects: centre must be below this
MAX_WALL_OBJ_Y       = 4.0    # wall objects: centre must be below this (skip ceiling fixtures)
MAX_FLOOR_VARIANTS  = 3   # max floor-object variants per scene
MAX_WALL_VARIANTS   = 2   # max wall-object variants per scene

# Wall pair aspect-ratio matching:
#   Two wall objects are "same shape" if their width/height ratios are within this
#   relative tolerance.  E.g. 0.15 = allow up to 15% relative difference.
WALL_AR_MATCH_THRESHOLD = 0.20

# Clean camera: equilateral triangle on perpendicular bisector
CLEAN_DIST_RANGE     = np.arange(2.0, 7.0, 0.5)  # fallback range (m from midpoint)
MAX_CLEAN_FRAMES     = 2

# Distorted: camera placed on the line (clean_cam → near_obj), at these
# fixed distances from the near object (tries from far to close).
LATERAL_OFFSETS  = [1.0, 1.5, 0.75, 2.0, 0.5, 2.5]  # metres to slide along AB axis
MIN_DIST_RATIO   = 1.2   # d_far / d_near must exceed this (meaningful illusion)
MAX_DIST_FRAMES        = 1

# Minimum separation between objects for illusion to work
MIN_SEPARATION_FLOOR = 2.5    # metres (floor objects — furniture needs more space)
MIN_SEPARATION_WALL  = 1.2    # metres (wall objects — pictures/shelves can be closer)

FOV_CHECK_DOT        = 0.65   # ~cos(49°) — slight margin beyond 90° FOV edges
CAMERA_HEIGHT        = 1.2
GRID_STEP            = 0.25
IMAGE_W              = 1024
IMAGE_H              = 1024

CLEAN_LAYOUT_ROOT    = "/Users/zhangyue/Desktop/Holodeck/clean_scene_layout/"
OUTPUT_BASE          = "/Users/zhangyue/Desktop/Holodeck/distortion_scenes_layout/"

_SKIP_JSON = {
    "target_subgraph.json", "object_attributes.json",
    "camera_poses.json", "structure_proxy.json", "occlusion_meta.json",
}

# ── math helpers ───────────────────────────────────────────────────────────────

def _v(p): return np.array([float(p["x"]), 0.0, float(p["z"])], dtype=float)
def _norm(v):
    n = np.linalg.norm(v); return v / n if n > 1e-8 else v
def _snap(v): return round(v / GRID_STEP) * GRID_STEP

def _rot_y_from_dir(dx, dz):
    return math.degrees(math.atan2(dx, dz)) % 360

def _wall_angle_diff(o1: dict, o2: dict) -> float:
    """
    Absolute angular difference (0–180°) between two wall objects' facing directions.
      ≈   0° → same wall (both face the same direction)
      ≈  90° → adjacent / connected walls (corner scenario)
      ≈ 180° → opposite walls (camera cannot easily see both simultaneously)
    """
    ry1 = float((o1.get("rotation") or {}).get("y", 0))
    ry2 = float((o2.get("rotation") or {}).get("y", 0))
    diff = abs(ry1 - ry2) % 360
    if diff > 180:
        diff = 360 - diff
    return diff   # in [0, 180]

def _left_right(cam_xz, rot_y, pos_a_xz, pos_b_xz):
    """
    Return {"large": "left"/"right", "small": "left"/"right"} where
    pos_a = large, pos_b = small.
    Right-hand rule: camera right = (cos rot_y, -sin rot_y) in XZ.
    """
    right = np.array([math.cos(math.radians(rot_y)),
                      -math.sin(math.radians(rot_y))])
    def side(pos):
        d = float(np.dot(_norm(pos - cam_xz), right))
        return "right" if d >= 0 else "left"
    return {"large": side(pos_a_xz), "small": side(pos_b_xz)}

def _room_bboxes(scene) -> list[tuple[float, float, float, float, str]]:
    """Return (mn_x, mx_x, mn_z, mx_z, room_id) for each room."""
    boxes = []
    for room in scene.get("rooms") or []:
        fp = room.get("floorPolygon", [])
        if not fp:
            continue
        rid = room.get("id", "")
        xs = [p["x"] for p in fp]; zs = [p["z"] for p in fp]
        boxes.append((min(xs), max(xs), min(zs), max(zs), rid))
    if not boxes:
        boxes = [(-10.0, 10.0, -10.0, 10.0, "")]
    return boxes

def _which_room(scene, x, z) -> str | None:
    """Return the room id that contains (x,z), or None if outside all rooms."""
    for mn_x, mx_x, mn_z, mx_z, rid in _room_bboxes(scene):
        if mn_x <= x <= mx_x and mn_z <= z <= mx_z:
            return rid
    return None

def _scene_bounds(scene):
    boxes = _room_bboxes(scene)
    all_mn_x = min(b[0] for b in boxes)
    all_mx_x = max(b[1] for b in boxes)
    all_mn_z = min(b[2] for b in boxes)
    all_mx_z = max(b[3] for b in boxes)
    return all_mn_x, all_mx_x, all_mn_z, all_mx_z

def _in_room(scene, x, z, margin=0.4):
    """True if (x,z) falls inside ANY room's bounding box (with margin)."""
    for mn_x, mx_x, mn_z, mx_z, _ in _room_bboxes(scene):
        if mn_x+margin <= x <= mx_x-margin and mn_z+margin <= z <= mx_z-margin:
            return True
    return False

def _apparent_ar(obj: dict, cam_xz: np.ndarray, true_ar: float) -> float:
    """
    Apparent aspect ratio of a wall-mounted object when viewed from cam_xz.

    The 'width' of the picture is foreshortened by cos(φ), where φ is the angle
    between the camera-to-object ray and the object's outward face normal.
      φ = 0°  → straight-on view  → apparent_AR = true_AR
      φ = 90° → edge-on view      → apparent_AR = 0
    """
    rot_y = float((obj.get("rotation") or {}).get("y", 0))
    # Outward face normal of the wall art in the XZ plane
    face_nx = math.sin(math.radians(rot_y))
    face_nz = math.cos(math.radians(rot_y))
    face_normal = np.array([face_nx, face_nz])

    obj_xz = np.array([float(obj["position"]["x"]), float(obj["position"]["z"])])
    obj_to_cam = cam_xz - obj_xz
    dist = float(np.linalg.norm(obj_to_cam))
    if dist < 1e-6:
        return true_ar
    obj_to_cam_unit = obj_to_cam / dist

    # cos(φ) = dot(obj→cam unit, face normal)  (clamped to [0,1])
    cos_phi = float(np.dot(obj_to_cam_unit, face_normal))
    cos_phi = max(0.0, min(1.0, cos_phi))

    # Apparent width shrinks by cos(φ); height is unchanged
    return true_ar * cos_phi


def _both_in_fov(cam_xz, rot_y, pos_a_xz, pos_b_xz):
    fwd = np.array([math.sin(math.radians(rot_y)), math.cos(math.radians(rot_y))])
    for p in (pos_a_xz, pos_b_xz):
        to = p - cam_xz
        if np.linalg.norm(to) < 1e-4: continue
        if float(np.dot(_norm(to), fwd)) < FOV_CHECK_DOT:
            return False
    return True

# ── object helpers ─────────────────────────────────────────────────────────────

def _base_type(obj):
    name = obj.get("object_name") or ""
    if name:
        base = re.sub(r"[-_]\d+$", "", str(name)).strip()
        if base: return base.lower()
    t = obj.get("objectType") or obj.get("type") or ""
    if t: return t.strip().lower()
    raw = obj.get("id") or ""
    return re.split(r"[|_\-]", str(raw))[0].strip().lower()

def _human_type(raw: str) -> str:
    return re.sub(r"[_]+", " ", str(raw)).strip()

def _footprint(obj, scene) -> float:
    """Largest horizontal dimension (m) — used for corridor-clearing width."""
    aabb = obj.get("axisAlignedBoundingBox") or {}
    sz   = aabb.get("size") or {}
    lx, lz = float(sz.get("x", 0)), float(sz.get("z", 0))
    if max(lx, lz) > 1e-4:
        return max(lx, lz)
    # fallback: object_selection_plan
    room_id  = obj.get("roomId", "")
    obj_name = re.sub(r"-\d+$", "", (obj.get("object_name") or obj.get("id", "")).split("(")[0].strip())
    plan_sz  = ((scene.get("object_selection_plan") or {})
                .get(room_id, {}).get(obj_name, {}).get("size"))
    if isinstance(plan_sz, list) and len(plan_sz) >= 2:
        return max(plan_sz[0], plan_sz[1]) / 100.0
    return 0.0


def _obj_size(obj, scene, category: str) -> float:
    """
    Visual size metric for size-contrast comparison between two same-type objects.

    Floor objects: max(x, z)  — horizontal footprint
    Wall objects : max(x, y, z) — wall art / shelves are thin in depth (z≈0),
                   so this captures the larger of width (x) and height (y).

    Falls back to object_selection_plan, then to the 'scale' field.
    """
    aabb = obj.get("axisAlignedBoundingBox") or {}
    sz   = aabb.get("size") or {}
    lx = float(sz.get("x", 0))
    ly = float(sz.get("y", 0))
    lz = float(sz.get("z", 0))

    if category == "wall":
        val = max(lx, ly, lz)
    else:
        val = max(lx, lz)

    if val > 1e-4:
        return val

    # fallback: object_selection_plan (mainly populated for floor objects)
    room_id  = obj.get("roomId", "")
    obj_name = re.sub(r"-\d+$", "", (obj.get("object_name") or obj.get("id", "")).split("(")[0].strip())
    plan_sz  = ((scene.get("object_selection_plan") or {})
                .get(room_id, {}).get(obj_name, {}).get("size"))
    if isinstance(plan_sz, list) and len(plan_sz) >= 2:
        return max(plan_sz[0], plan_sz[1]) / 100.0

    # last resort: scale field (relative, but still lets us compare two instances)
    scale = obj.get("scale") or {}
    sx = float(scale.get("x", 0))
    sy = float(scale.get("y", 0))
    sz_ = float(scale.get("z", 0))
    if category == "wall":
        return max(sx, sy, sz_)
    return max(sx, sz_)

def _wall_dims(obj, scene) -> tuple[float, float] | None:
    """
    Physical face dimensions (d_large_cm, d_small_cm) of a wall-mounted object in cm.
    d_large ≥ d_small; depth (thickness) is excluded.

    Priority:
      1. axisAlignedBoundingBox.size  (usually absent in Holodeck JSONs)
      2. object_selection_plan size array — depth identified as the smallest value.
    Returns None if no dimensions are available.
    """
    # 1. Bounding box
    aabb = obj.get("axisAlignedBoundingBox") or {}
    sz   = aabb.get("size") or {}
    lx = float(sz.get("x", 0))
    ly = float(sz.get("y", 0))
    lz = float(sz.get("z", 0))
    w, h = max(lx, lz), ly
    if w > 1e-4 and h > 1e-4:
        # convert metres → cm
        return (round(max(w, h) * 100, 1), round(min(w, h) * 100, 1))

    # 2. object_selection_plan
    obj_id = obj.get("id", "") or obj.get("object_name", "")
    m = re.search(r'\(([^)]+)\)', obj_id)
    room_key = m.group(1).strip() if m else ""
    obj_name = re.sub(r"-\d+$", "", obj_id.split("(")[0].strip())
    plan = ((scene.get("object_selection_plan") or {})
            .get(room_key, {}).get(obj_name, {}))
    plan_sz = plan.get("size")
    if isinstance(plan_sz, list) and len(plan_sz) >= 3:
        dims = sorted([float(plan_sz[0]), float(plan_sz[1]), float(plan_sz[2])])
        # dims[0] = depth (smallest), dims[1] and dims[2] = face dimensions
        if dims[1] > 1e-4:
            return (dims[2], dims[1])   # (large_cm, small_cm), already in cm from plan
    elif isinstance(plan_sz, list) and len(plan_sz) == 2:
        d0, d1 = float(plan_sz[0]), float(plan_sz[1])
        if min(d0, d1) > 1e-4:
            return (max(d0, d1), min(d0, d1))

    return None


def _wall_aspect_ratio(obj, scene) -> float | None:
    """
    Aspect ratio (max face dimension / min face dimension) of a wall-mounted object.
    Always ≥ 1.  Delegates to _wall_dims.
    """
    dims = _wall_dims(obj, scene)
    if dims is None:
        return None
    large, small = dims
    return large / small if small > 1e-4 else None


def _floor_aspect_ratio(obj, scene) -> float | None:
    """
    Footprint aspect ratio (max_horizontal / min_horizontal, always ≥ 1) of a floor object.
    Used to check whether two perpendicular-facing objects have similar footprint shapes
    — a prerequisite for a meaningful floor-shape question.

    Uses axisAlignedBoundingBox lx / lz first; falls back to object_selection_plan.
    """
    aabb = obj.get("axisAlignedBoundingBox") or {}
    sz   = aabb.get("size") or {}
    lx = float(sz.get("x", 0))
    lz = float(sz.get("z", 0))
    if lx > 1e-4 and lz > 1e-4:
        return max(lx, lz) / min(lx, lz)

    # Fallback: object_selection_plan.  For floor objects [w, h, d] or [w, d, h]
    # — height (vertical extent) is typically the MIDDLE dimension for a table/chair,
    # or the LARGEST for a tall cabinet.  We identify height as the dimension that
    # is most different from the other two; the remaining two are the footprint.
    # Simplest reliable heuristic: sort all three, take the two SMALLEST as footprint dims.
    room_id  = obj.get("roomId", "")
    obj_name = re.sub(r"-\d+$", "", (obj.get("object_name") or obj.get("id", "")).split("(")[0].strip())
    plan_sz  = ((scene.get("object_selection_plan") or {})
                .get(room_id, {}).get(obj_name, {}).get("size"))
    if isinstance(plan_sz, list) and len(plan_sz) >= 3:
        dims = sorted([float(plan_sz[0]), float(plan_sz[1]), float(plan_sz[2])])
        # dims[0] ≤ dims[1] ≤ dims[2]; for a floor object the two smallest are
        # often depth + width (footprint); the largest is often height
        d0, d1 = dims[0], dims[1]
        if d0 > 1e-4:
            return d1 / d0    # always ≥ 1
    return None


def _obj_category(obj) -> str:
    """'floor' if centre y <= MAX_FLOOR_OBJ_Y, else 'wall'."""
    y = float(obj.get("position", {}).get("y", 99))
    return "floor" if y <= MAX_FLOOR_OBJ_Y else "wall"

# ── occlusion clearing ─────────────────────────────────────────────────────────

CLEAR_CORRIDOR_WIDTH = 0.6   # metres — half-width of line-of-sight corridor to clear

def _pt_in_segment_corridor(p: np.ndarray, a: np.ndarray, b: np.ndarray, width: float) -> bool:
    """True if 2-D point p is within `width` metres of segment [a, b] (and between a and b)."""
    ab     = b - a
    ab_len = float(np.linalg.norm(ab))
    if ab_len < 1e-8:
        return float(np.linalg.norm(p - a)) < width
    t = float(np.dot(p - a, ab)) / (ab_len ** 2)
    if t < 0.0 or t > 1.0:
        return False
    closest = a + t * ab
    return float(np.linalg.norm(p - closest)) < width


def _clear_blocking_objects(scene_mod: dict,
                             obj_a: dict, obj_b: dict,
                             camera_xz_list: list[np.ndarray],
                             pair_cat: str) -> int:
    """
    Remove floor objects (other than obj_a / obj_b) whose centre lies inside any
    line-of-sight corridor:
      • obj_a ↔ obj_b  (between the two targets)
      • each camera ↔ obj_a
      • each camera ↔ obj_b

    Wall objects are never removed (they don't obstruct the ground-level view).
    Returns the number of objects removed.
    """
    target_ids = {obj_a["id"], obj_b["id"]}
    pa = np.array([float(obj_a["position"]["x"]), float(obj_a["position"]["z"])])
    pb = np.array([float(obj_b["position"]["x"]), float(obj_b["position"]["z"])])

    # Only clear objects blocking camera → target sightlines (not between targets)
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for cam in camera_xz_list:
        segments.append((cam, pa))
        segments.append((cam, pb))

    remove_ids: set[str] = set()
    for obj in scene_mod.get("objects", []):
        oid = obj["id"]
        if oid in target_ids or oid in remove_ids:
            continue
        y = float(obj.get("position", {}).get("y", 99))
        if y > MAX_FLOOR_OBJ_Y:
            continue   # don't remove wall / ceiling objects
        ox = float(obj["position"]["x"])
        oz = float(obj["position"]["z"])
        op = np.array([ox, oz])
        fp = _footprint(obj, scene_mod)
        w  = CLEAR_CORRIDOR_WIDTH + fp / 2.0   # wider corridor for bigger objects
        for a, b in segments:
            if _pt_in_segment_corridor(op, a, b, w):
                remove_ids.add(oid)
                break

    if remove_ids:
        scene_mod["objects"] = [o for o in scene_mod["objects"]
                                 if o["id"] not in remove_ids]
    return len(remove_ids)


def _camera_height_for(obj_a, obj_b) -> float:
    """For wall objects, raise camera to roughly the objects' mid-height."""
    ya = float(obj_a.get("position", {}).get("y", 0))
    yb = float(obj_b.get("position", {}).get("y", 0))
    avg_y = (ya + yb) / 2.0
    if avg_y > MAX_FLOOR_OBJ_Y:
        return round(min(avg_y, 2.2), 2)   # look at wall objects at their height
    return CAMERA_HEIGHT

def _clone_object(orig: dict, new_id: str, pos_x: float, pos_z: float,
                  scale: float) -> dict:
    """Deep-copy orig, give it a new id/name, place at (pos_x, pos_z), apply scale."""
    clone = copy.deepcopy(orig)
    # adjust y: centre height scales with object
    orig_y = float(orig["position"]["y"])
    clone["position"] = {"x": pos_x, "y": round(orig_y * scale, 4), "z": pos_z}
    clone["id"]          = new_id
    clone["object_name"] = new_id
    clone["scale"]       = {"x": scale, "y": scale, "z": scale}
    # clear vertices (optional layout hint, not needed for rendering)
    clone["vertices"] = []
    return clone

# ── camera samplers ────────────────────────────────────────────────────────────

def _find_clean_camera(scene, pos_large_xz, pos_small_xz, extra_margin=0.5):
    """
    Find a clean camera on the perpendicular bisector, equidistant from both
    objects.  Prefer closer distances (both objects clearly visible and large
    in frame) as long as both fit within the 90° FOV.

    Minimum valid perp distance from midpoint = sep/2  (objects at exactly 45°).
    We start just above that and sweep outward.
    extra_margin: additional metres beyond sep/2 to start (use smaller for wall objs).
    """
    mid  = (pos_large_xz + pos_small_xz) / 2.0
    sep  = float(np.linalg.norm(pos_large_xz - pos_small_xz))
    ab   = pos_small_xz - pos_large_xz
    perp = _norm(np.array([-ab[1], ab[0]]))

    # Start from just above sep/2 (minimum for 45° FOV), go outward
    min_dist = sep / 2.0 + extra_margin
    dist_candidates = list(np.arange(min_dist, min_dist + 4.0, 0.25))

    for dist in dist_candidates:
        for side in [1, -1]:
            cx = _snap(mid[0] + side * perp[0] * dist)
            cz = _snap(mid[1] + side * perp[1] * dist)
            if not _in_room(scene, cx, cz): continue
            cam = np.array([cx, cz])
            ry  = _rot_y_from_dir(mid[0] - cx, mid[1] - cz)
            if not _both_in_fov(cam, ry, pos_large_xz, pos_small_xz): continue
            return {
                "x": float(cx), "z": float(cz), "rot_y": float(ry),
                "dist": float(dist),
            }
    return None


def _slide_lateral(scene, clean_xz, pos_a_xz, pos_b_xz):
    """
    Translate the clean camera ALONG the AB axis (sideways) by various offsets.
    This keeps the camera at roughly the same distance from the scene — both
    objects stay fully visible — while one becomes closer than the other,
    creating the size-distance illusion without a close-up of either object.

    Returns a list of up to 2 dicts: one with A as the near object, one with B.
    Each dict: {x, z, rot_y, dist_near, dist_far, near_obj ("a"/"b")}
    """
    ab_unit = _norm(pos_b_xz - pos_a_xz)   # unit vector A→B

    offsets = LATERAL_OFFSETS

    results: dict[str, list] = {"a": [], "b": []}  # near_obj → list (up to MAX_DIST_FRAMES each)
    for offset in offsets:
        for sign, near_obj in [(+1, "b"), (-1, "a")]:
            # Sliding +AB makes camera closer to B; sliding -AB makes closer to A
            if len(results[near_obj]) >= MAX_DIST_FRAMES:
                continue   # already have enough for this side
            new_cam = clean_xz + sign * ab_unit * offset
            cx = _snap(float(new_cam[0]))
            cz = _snap(float(new_cam[1]))
            if not _in_room(scene, cx, cz): continue
            cam = np.array([cx, cz])

            to_a = _norm(pos_a_xz - cam)
            to_b = _norm(pos_b_xz - cam)
            bisect = to_a + to_b
            bn = float(np.linalg.norm(bisect))
            if bn < 1e-4: continue
            bisect = bisect / bn

            if float(np.dot(to_a, bisect)) < FOV_CHECK_DOT: continue
            if float(np.dot(to_b, bisect)) < FOV_CHECK_DOT: continue

            d_a = float(np.linalg.norm(pos_a_xz - cam))
            d_b = float(np.linalg.norm(pos_b_xz - cam))
            d_near = min(d_a, d_b)
            d_far  = max(d_a, d_b)
            actual_near = "a" if d_a <= d_b else "b"
            if actual_near != near_obj:
                continue   # sanity check — offset should make the expected side closer
            if d_far / d_near < MIN_DIST_RATIO:
                continue

            ry = _rot_y_from_dir(float(bisect[0]), float(bisect[1]))
            results[near_obj].append({
                "x": float(cx), "z": float(cz), "rot_y": float(ry),
                "dist_near": d_near, "dist_far": d_far,
                "near_obj": near_obj,
            })

    # Return interleaved: a[0], b[0], a[1], b[1] — so frame order is
    # left-close, right-close, left-far, right-far (sorted by offset magnitude)
    merged = []
    for i in range(MAX_DIST_FRAMES):
        if i < len(results["a"]): merged.append(results["a"][i])
        if i < len(results["b"]): merged.append(results["b"][i])
    return merged

# ── save variant ───────────────────────────────────────────────────────────────

def _safe(s): return re.sub(r"[^a-zA-Z0-9_]", "_", s)

def _build_frame_entry(fr: dict, cam_height: float,
                       distortion_sub_type: str,
                       ar_a: float | None, ar_b: float | None) -> dict:
    """
    Build a single frame dict for frame_meta.json.
    Adds a 'shape_info' block for wall/shape pairs and a 'size_info' block for
    floor/size pairs, recording what the model should perceive vs. reality.
    """
    scene_type  = fr["scene_type"]          # "clean" | "distorted"
    near_object = fr.get("near_object")     # None | "a" | "b"

    base = {
        "frame_idx":      fr["frame_idx"],
        "image_file":     fr["image_file"],
        "scene_type":     scene_type,
        "near_object":    near_object,
        "larger_appears": fr.get("larger_appears"),
        "a_side":         fr.get("a_side"),
        "b_side":         fr.get("b_side"),
        "description":    fr["description"],
        "camera": {
            "position":     {"x": fr["x"], "y": cam_height, "z": fr["z"]},
            "rotation_yaw": fr["rot_y"],
        },
    }

    if distortion_sub_type == "shape":
        # Wall pair: track which object is foreshortened and apparent shape match.
        # The near object is viewed more straight-on → natural AR.
        # The far object is at a steeper angle → width compressed → appears different AR.
        if scene_type == "clean":
            foreshortened_obj    = None
            apparent_shapes_match = True   # symmetric camera → both look same shape
        else:
            # far object = the one that is NOT near_object
            foreshortened_obj    = ("a" if near_object == "b" else "b") if near_object else None
            apparent_shapes_match = False  # one appears foreshortened → shapes look different

        base["shape_info"] = {
            "true_ar_a":             round(ar_a, 3) if ar_a else None,
            "true_ar_b":             round(ar_b, 3) if ar_b else None,
            "shapes_match_in_reality": True,   # we only select same-AR pairs
            "foreshortened_obj":     foreshortened_obj,   # None / "a" / "b"
            "apparent_shapes_match": apparent_shapes_match,
            # Note: apparent_ar_* fields reflect the projected AR from this camera.
            # For clean frames both should be close (enforced at generation time).
            # For distorted frames one will be compressed relative to the other.
            # Human-readable gloss for question construction:
            "gt_note": (
                "Both objects have the same true aspect ratio; "
                + ("camera is symmetric so both appear the same shape."
                   if scene_type == "clean"
                   else f"obj-{foreshortened_obj} is viewed at an oblique angle and "
                        f"appears compressed — shapes look different despite being equal.")
            ),
        }

    else:
        # Floor pair: track which object appears larger and apparent size match.
        if scene_type == "clean":
            appears_larger_obj   = None
            apparent_sizes_match = True
        else:
            appears_larger_obj   = near_object  # nearer → appears larger
            apparent_sizes_match = False

        base["size_info"] = {
            "sizes_match_in_reality": True,   # same assetId → same physical size
            "appears_larger_obj":     appears_larger_obj,   # None / "a" / "b"
            "apparent_sizes_match":   apparent_sizes_match,
            "gt_note": (
                "Both objects are physically identical (same 3-D model); "
                + ("camera is equidistant so both appear the same size."
                   if scene_type == "clean"
                   else f"obj-{appears_larger_obj} is nearer and appears larger "
                        f"due to perspective — sizes look different despite being equal.")
            ),
        }

    return base


def _save_variant(json_path, scene_folder, scene_mod,
                  obj_a, obj_b, type_name, frames, cam_height=CAMERA_HEIGHT,
                  pair_cat="floor", ar_a=None, ar_b=None, angle_diff=0.0):
    safe_type = _safe(type_name)
    name      = f"distortion_sizedistance_{safe_type}"
    # avoid collision if multiple variants of same type
    base_out  = Path(OUTPUT_BASE) / scene_folder
    idx = 0
    while (base_out / f"{name}_{idx:02d}").exists():
        idx += 1
    out_dir = base_out / f"{name}_{idx:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write modified scene
    (out_dir / "scene.json").write_text(
        json.dumps(scene_mod, indent=2, ensure_ascii=False), encoding="utf-8")

    # camera_poses.json
    cam_frames = []
    for fr in frames:
        cam_frames.append({
            "frame_idx":   fr["frame_idx"],
            "image_file":  fr["image_file"],
            "scene_type":  fr["scene_type"],
            "near_object": fr.get("near_object"),
            "larger_appears": fr.get("larger_appears"),
            "position":    {"x": fr["x"], "y": cam_height, "z": fr["z"]},
            "rotation":    {"x": 0.0, "y": fr["rot_y"], "z": 0.0},
            "fov":         90.0,
        })
    cam_data = {
        "primary_sequence": "multiview",
        "camera_frame_for_modality_eval": "multiview",
        "multiview": {
            "type": "multiview", "n_views": len(cam_frames),
            "image_width": IMAGE_W, "image_height": IMAGE_H,
            "frames": cam_frames,
        },
    }
    (out_dir / "camera_poses.json").write_text(
        json.dumps(cam_data, indent=2), encoding="utf-8")

    n_clean = sum(1 for f in frames if f["scene_type"] == "clean")
    n_dist  = sum(1 for f in frames if f["scene_type"] == "distorted")

    # Distortion sub-type:
    #   wall        → always "shape"  (same-AR pictures; oblique angle compresses one)
    #   floor ~0°   → "size"          (both face same way; lateral shift makes one appear bigger)
    #   floor ~90°  → "shape"         (perpendicular; lateral shift makes one appear narrower)
    #   floor ~180° → filtered earlier (opposite facing; skip)
    if pair_cat == "wall":
        distortion_sub_type = "shape"
    elif 45 < angle_diff < 135:
        distortion_sub_type = "shape"   # perpendicular floor objects
    else:
        distortion_sub_type = "size"    # parallel floor objects

    if distortion_sub_type == "shape":
        if pair_cat == "wall":
            layout_str  = "corner" if 45 < angle_diff < 135 else "same-wall"
            layout_key  = "wall_layout"
        else:
            layout_str  = "perpendicular"
            layout_key  = "floor_layout"

        # Physical face dimensions for wall objects (large_cm × small_cm, depth excluded)
        dims_a = _wall_dims(obj_a, scene_mod)
        dims_b = _wall_dims(obj_b, scene_mod)

        def _fmt_dims(d):
            return {"large_cm": d[0], "small_cm": d[1]} if d else None

        extra_meta: dict = {
            layout_key:        layout_str,
            "angle_diff":      round(angle_diff, 1),
            "aspect_ratio_a":  round(ar_a, 3) if ar_a is not None else None,
            "aspect_ratio_b":  round(ar_b, 3) if ar_b is not None else None,
            "ar_avg":          round((ar_a + ar_b) / 2, 3) if (ar_a and ar_b) else None,
            "dims_a":          _fmt_dims(dims_a),   # {"large_cm": X, "small_cm": Y}
            "dims_b":          _fmt_dims(dims_b),
        }
        ar_str = f"AR≈{ar_a:.2f}" if ar_a else "AR=?"
        summary = (
            f"{type_name} [{pair_cat}/{layout_str}, shape-distortion, "
            f"angle={angle_diff:.0f}°, {ar_str}]: "
            f"two same-shape objects. "
            f"{n_clean} clean (symmetric) + {n_dist} distorted (oblique angle) frames."
        )
    else:
        # Floor size pair: same-model objects; lateral shift creates size illusion
        # Save footprint dims for reference (lx × lz from bounding box or plan)
        def _floor_dims(obj):
            aabb = obj.get("axisAlignedBoundingBox") or {}
            sz   = aabb.get("size") or {}
            lx = float(sz.get("x", 0)) * 100   # m → cm
            lz = float(sz.get("z", 0)) * 100
            if lx > 1e-2 and lz > 1e-2:
                return {"lx_cm": round(lx, 1), "lz_cm": round(lz, 1)}
            room_id  = obj.get("roomId", "")
            obj_name = re.sub(r"-\d+$", "", (obj.get("object_name") or obj.get("id", "")).split("(")[0].strip())
            plan_sz  = ((scene_mod.get("object_selection_plan") or {})
                        .get(room_id, {}).get(obj_name, {}).get("size"))
            if isinstance(plan_sz, list) and len(plan_sz) >= 2:
                return {"plan_size_cm": [round(float(v), 1) for v in plan_sz]}
            return None

        extra_meta = {
            "dims_a": _floor_dims(obj_a),
            "dims_b": _floor_dims(obj_b),
        }
        summary = (
            f"{type_name} [floor, size-distortion, angle={angle_diff:.0f}°]: "
            f"two same-model objects (A & B). "
            f"{n_clean} clean (equidistant) + {n_dist} distorted (lateral shift) frames."
        )

    frame_meta = {
        "distortion_type":     "size_distance",
        "distortion_sub_type": distortion_sub_type,
        "object_type":         type_name,
        "pair_category":       pair_cat,     # "floor" | "wall"
        "object_a_id":         obj_a["id"],
        "object_b_id":         obj_b["id"],
        "n_clean_frames":      n_clean,
        "n_distorted_frames":  n_dist,
        "summary":             summary,
        **extra_meta,
        "frames": [
            _build_frame_entry(fr, cam_height, distortion_sub_type, ar_a, ar_b)
            for fr in frames
        ],
    }
    (out_dir / "frame_meta.json").write_text(
        json.dumps(frame_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"    → {out_dir.name}/  ({n_clean} clean + {n_dist} dist = {len(frames)} frames)")
    return True

# ── per-scene generator ────────────────────────────────────────────────────────

def generate_sizedistance_for_scene(json_path: str, scene_folder: str) -> int:
    try:
        scene = json.loads(Path(json_path).read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        print(f"  [ERROR] {e}"); return 0

    objects = scene.get("objects", [])

    # Skip structural / non-object types.
    # Note: "wall" alone skips the wall structure, but NOT wall-mounted objects
    # like "wall_art", "wall_shelf", "metal_wall_shelf" — those are fine.
    _SKIP_TYPES = ("ceiling", "floor", "window", "door",
                   "light", "lamp", "vent", "outlet", "switch", "baseboard")
    _SKIP_EXACT = {"wall"}   # exact base-type match only

    # Group objects by base type — floor AND wall-mounted (pictures, shelves, etc.)
    by_type: dict[str, list] = {}
    for obj in objects:
        y = float(obj.get("position", {}).get("y", 99))
        # floor objects: y <= MAX_FLOOR_OBJ_Y
        # wall objects: MAX_FLOOR_OBJ_Y < y <= MAX_WALL_OBJ_Y
        if y > MAX_WALL_OBJ_Y:
            continue
        fp = _footprint(obj, scene)
        if fp < MIN_OBJ_SIZE:
            continue
        t = _base_type(obj)
        if t in _SKIP_EXACT:
            continue
        if any(t == x or t.startswith(x + "_") or t.endswith("_" + x)
               for x in _SKIP_TYPES):
            continue
        # tag whether it's a floor or wall object for later use
        category = "floor" if y <= MAX_FLOOR_OBJ_Y else "wall"
        # Re-compute size with category-aware metric (wall objects use max(x,y,z))
        size_val = _obj_size(obj, scene, category)
        if size_val < MIN_OBJ_SIZE:
            size_val = fp   # keep original if category-aware also too small
        by_type.setdefault(t, []).append((obj, size_val, category))

    # Keep types that have at least 2 instances in the same room.
    # Floor pairs only use floor objects; wall pairs only use wall objects.
    # For each type+category group, collect all (n choose 2) candidate pairs
    # in the SAME room, sorted appropriately:
    #   floor → sorted by XZ separation (largest first)
    #   wall  → sorted by footprint ratio DESC (most contrasting pair first),
    #            then by sep DESC as tiebreaker.
    # pairs element: (type_name, obj_a, obj_b, cat, fp_ratio)
    #   obj_a always has the LARGER footprint of the two.
    pairs = []
    for t, objs in by_type.items():
        floor_objs = [(o, fp) for o, fp, cat in objs if cat == "floor"]
        wall_objs  = [(o, fp) for o, fp, cat in objs if cat == "wall"]
        for group, cat in [(floor_objs, "floor"), (wall_objs, "wall")]:
            if len(group) < 2:
                continue
            candidates = []
            for (o1, fp1), (o2, fp2) in itertools.combinations(group, 2):
                p1 = np.array([float(o1["position"]["x"]), float(o1["position"]["z"])])
                p2 = np.array([float(o2["position"]["x"]), float(o2["position"]["z"])])
                # Same-room check
                rid1 = _which_room(scene, float(p1[0]), float(p1[1]))
                rid2 = _which_room(scene, float(p2[0]), float(p2[1]))
                if rid1 is not None and rid2 is not None and rid1 != rid2:
                    continue   # different rooms — skip

                sep_val = float(np.linalg.norm(p1 - p2))

                if cat == "floor":
                    # Floor: same assetId = same 3D model = same physical shape/size
                    aid1 = o1.get("assetId"); aid2 = o2.get("assetId")
                    if aid1 and aid2 and aid1 != aid2:
                        continue

                    # Check facing-direction difference:
                    #   ~0°   → size question (both face same way, lateral shift = size illusion)
                    #   ~90°  → shape question (perpendicular; lateral shift = shape illusion)
                    #   ~180° → opposite — skip (camera can't simultaneously face both)
                    ad = _wall_angle_diff(o1, o2)
                    if ad > 150:
                        continue   # opposite facing — skip

                    # For shape pairs, compute footprint AR for matching check
                    if 45 < ad < 135:
                        ar1 = _floor_aspect_ratio(o1, scene)
                        ar2 = _floor_aspect_ratio(o2, scene)
                    else:
                        ar1, ar2 = None, None

                    # Sort key: prefer shape (perpendicular) pairs, then larger sep
                    is_shape = 1 if 45 < ad < 135 else 0
                    candidates.append((sep_val, is_shape, o1, o2, 1.0, ar1, ar2, round(ad, 1)))

                else:  # cat == "wall"
                    # Wall pairs: allow different assetIds (different picture content is fine).
                    # Same-type objects share the same planned size in object_selection_plan,
                    # so their aspect ratios should be approximately equal by design.

                    # Filter by wall orientation:
                    #   same wall    (~0°)  → OK: side-by-side on same wall
                    #   adjacent     (~90°) → OK: corner scenario (preferred — creates
                    #                              natural foreshortening on one picture)
                    #   opposite     (~180°)→ SKIP: camera can't face both simultaneously
                    angle_diff = _wall_angle_diff(o1, o2)
                    if angle_diff > 150:
                        continue   # opposite walls — skip

                    # Soft AR check using planned sizes (informational, not hard filter)
                    ar1 = _wall_aspect_ratio(o1, scene)
                    ar2 = _wall_aspect_ratio(o2, scene)
                    if ar1 and ar2:
                        ar_diff_rel = abs(ar1 - ar2) / max(ar1, ar2)
                        if ar_diff_rel > WALL_AR_MATCH_THRESHOLD:
                            continue   # plan ARs are too far apart — skip this pair
                    ar_avg = ((ar1 + ar2) / 2.0) if (ar1 and ar2) else ar1 or ar2

                    # Prefer adjacent-wall (corner) pairs over same-wall pairs:
                    # sort key: adjacent score (1 if ~90°, 0 if ~0°), then sep
                    is_corner = 1 if 45 < angle_diff < 135 else 0
                    candidates.append((sep_val, is_corner, o1, o2, ar_avg or 1.0, ar1, ar2,
                                       round(angle_diff, 1)))

            if cat == "wall":
                # Corner pairs first (is_corner=1), then by separation (largest first)
                candidates.sort(key=lambda x: (-x[1], -x[0]))
                for sep_v, is_corner, o_a, o_b, ar_avg, ar1, ar2, ad in candidates:
                    pairs.append((t, o_a, o_b, cat, ar_avg, ar1, ar2, ad))
            else:
                # Floor: parallel (same-direction, size question) first;
                # perpendicular (shape question) only as fallback if no parallel pair.
                # Within each group, prefer larger separation.
                candidates.sort(key=lambda x: (x[1], -x[0]))  # is_shape=0 before 1
                for sep_v, is_shape, o_a, o_b, extra, ar1, ar2, ad in candidates:
                    pairs.append((t, o_a, o_b, cat, extra, ar1, ar2, ad))

    if not pairs:
        print("  [SKIP] no same-type pairs found"); return 0

    saved       = 0
    saved_floor = 0
    saved_wall  = 0
    done_types: set[tuple] = set()   # (type_name, cat) pairs already successfully saved
    for type_name, obj_large, obj_small_orig, pair_cat, extra_val, ar_a, ar_b, angle_diff in pairs:
        pos_large = np.array([float(obj_large["position"]["x"]),
                               float(obj_large["position"]["z"])])
        pos_small = np.array([float(obj_small_orig["position"]["x"]),
                               float(obj_small_orig["position"]["z"])])

        # Skip if we already successfully saved a variant for this type+category
        if (type_name, pair_cat) in done_types:
            continue
        # Enforce per-category limits (floor and wall are counted separately)
        if pair_cat == "floor" and saved_floor >= MAX_FLOOR_VARIANTS:
            continue
        if pair_cat == "wall"  and saved_wall  >= MAX_WALL_VARIANTS:
            continue

        sep = float(np.linalg.norm(pos_large - pos_small))
        min_sep = MIN_SEPARATION_WALL if pair_cat == "wall" else MIN_SEPARATION_FLOOR
        if sep < min_sep:
            print(f"  [SKIP] {type_name} ({pair_cat}): objects too close ({sep:.2f}m < {min_sep}m)")
            continue

        # No resize — use the original scene as-is
        scene_mod = copy.deepcopy(scene)

        # ── Wall pairs: remove all OTHER same-type wall objects from the scene ──
        # Ensures the clean view shows ONLY the chosen pair, so their shared
        # aspect ratio (shape) is unambiguous.
        if pair_cat == "wall":
            keep_ids = {obj_large["id"], obj_small_orig["id"]}
            before_wall = len(scene_mod["objects"])
            scene_mod["objects"] = [
                o for o in scene_mod["objects"]
                if not (_base_type(o) == type_name and o["id"] not in keep_ids)
            ]
            n_wall_removed = before_wall - len(scene_mod["objects"])
            if n_wall_removed:
                print(f"    [WALL] removed {n_wall_removed} other '{type_name}' object(s) from scene")

        # ── Determine sub-type for this pair ─────────────────────────���─────
        if pair_cat == "wall" or (45 < angle_diff < 135):
            pair_sub_type = "shape"
        else:
            pair_sub_type = "size"

        # ── Sample cameras ────────────────────────────────��─────────────────
        frames    = []
        frame_idx = 0
        mid_xz    = (pos_large + pos_small) / 2.0
        human_type = _human_type(type_name)

        # 1. Find clean equilateral camera position
        # Wall objects can be closer together → use smaller extra_margin
        cam_margin = 0.3 if pair_cat == "wall" else 0.5
        clean_cam = _find_clean_camera(scene_mod, pos_large, pos_small, extra_margin=cam_margin)
        if clean_cam is None:
            print(f"  [SKIP] {type_name}: no clean camera"); continue

        ref_xz = np.array([clean_cam["x"], clean_cam["z"]])

        # ── Shape pairs: verify apparent ARs are similar from clean camera ───
        # For wall-corner or floor-perpendicular pairs the bisector camera may see
        # the two objects at different angles → they look different shapes already in
        # the "clean" view.  Skip if apparent ARs diverge too much.
        if pair_sub_type == "shape" and ar_a is not None and ar_b is not None:
            app_ar_a = _apparent_ar(obj_large,      ref_xz, ar_a)
            app_ar_b = _apparent_ar(obj_small_orig, ref_xz, ar_b)
            if app_ar_a > 1e-4 and app_ar_b > 1e-4:
                app_diff = abs(app_ar_a - app_ar_b) / max(app_ar_a, app_ar_b)
                if app_diff > WALL_AR_MATCH_THRESHOLD:
                    print(f"  [SKIP] {type_name} (wall): apparent ARs differ in clean view "
                          f"({app_ar_a:.2f} vs {app_ar_b:.2f}, diff={app_diff:.0%})")
                    continue

        clean_sides = _left_right(ref_xz, clean_cam["rot_y"], pos_large, pos_small)
        frames.append({
            "frame_idx":      frame_idx,
            "image_file":     f"view_{frame_idx:03d}.png",
            "scene_type":     "clean",
            "near_object":    None,
            "larger_appears": None,        # equidistant — no illusion
            "a_side":         clean_sides["large"],   # which side object A appears on
            "b_side":         clean_sides["small"],   # which side object B appears on
            "x": clean_cam["x"], "z": clean_cam["z"], "rot_y": clean_cam["rot_y"],
            "description": (
                f"Equidistant view d={clean_cam['dist']:.1f}m from midpoint. "
                f"Object A on {clean_sides['large']}, object B on {clean_sides['small']}. "
                f"Camera equidistant from both objects."
            ),
        })
        frame_idx += 1

        # 2 & 3. Distorted frames: translate camera LATERALLY along AB axis
        #    Sliding toward A side → A is nearer → A appears larger
        #    Sliding toward B side → B is nearer → B appears larger
        #    Camera stays at similar total distance; both objects remain fully visible.
        lateral_results = _slide_lateral(scene_mod, ref_xz, pos_large, pos_small)
        for lr in lateral_results:
            # lr["near_obj"] = "a" or "b"
            near_obj = lr["near_obj"]      # "a" or "b"
            far_obj  = "b" if near_obj == "a" else "a"
            cam_lr   = np.array([lr["x"], lr["z"]])
            sides_lr = _left_right(cam_lr, lr["rot_y"], pos_large, pos_small)
            # sides_lr["large"] = side of obj_a (larger footprint), ["small"] = side of obj_b
            a_side   = sides_lr["large"]
            b_side   = sides_lr["small"]
            near_side = a_side if near_obj == "a" else b_side
            far_side  = b_side if near_obj == "a" else a_side
            if pair_sub_type == "shape":
                dist_desc = (
                    f"Camera translated toward {near_obj.upper()}-side. "
                    f"Near (obj-{near_obj}): {lr['dist_near']:.1f}m on {near_side}. "
                    f"Far (obj-{far_obj}): {lr['dist_far']:.1f}m on {far_side}. "
                    f"Obj-{far_obj} viewed at oblique angle — appears foreshortened/narrower."
                )
            else:
                dist_desc = (
                    f"Camera translated toward {near_obj.upper()}-side. "
                    f"Near (obj-{near_obj}): {lr['dist_near']:.1f}m on {near_side}. "
                    f"Far (obj-{far_obj}): {lr['dist_far']:.1f}m on {far_side}. "
                    f"Obj-{near_obj} appears larger due to proximity."
                )
            frames.append({
                "frame_idx":      frame_idx,
                "image_file":     f"view_{frame_idx:03d}.png",
                "scene_type":     "distorted",
                "near_object":    near_obj,
                "larger_appears": near_obj if pair_sub_type == "size" else None,
                "a_side":         a_side,
                "b_side":         b_side,
                "x": lr["x"], "z": lr["z"], "rot_y": lr["rot_y"],
                "description":    dist_desc,
            })
            frame_idx += 1

        if not any(f["scene_type"] == "distorted" for f in frames):
            print(f"  [SKIP] {type_name}: no distorted cameras"); continue

        # ── Clear objects blocking line-of-sight (floor pairs only) ────────
        if pair_cat == "floor":
            all_cam_xz = [np.array([f["x"], f["z"]]) for f in frames]
            n_removed = _clear_blocking_objects(
                scene_mod, obj_large, obj_small_orig, all_cam_xz, pair_cat)
            if n_removed:
                print(f"    [CLEAR] removed {n_removed} blocking object(s)")

        cam_h = _camera_height_for(obj_large, obj_small_orig)
        # Determine sub_type for logging
        if pair_cat == "wall":
            sub_layout = "corner" if 45 < angle_diff < 135 else "same-wall"
        elif 45 < angle_diff < 135:
            sub_layout = "perpendicular"
        else:
            sub_layout = "parallel"
        ar_str = f"AR_a={ar_a:.2f}, AR_b={ar_b:.2f}" if (ar_a and ar_b) else ""
        print(f"  PAIR [{pair_cat}/{sub_layout}] {human_type}: "
              f"{obj_large['id']} ↔ {obj_small_orig['id']}  "
              f"(sep={sep:.1f}m, cam_y={cam_h}m, angle={angle_diff:.0f}°{', ' + ar_str if ar_str else ''})")
        _save_variant(json_path, scene_folder, scene_mod,
                      obj_large, obj_small_orig, human_type, frames, cam_height=cam_h,
                      pair_cat=pair_cat, ar_a=ar_a, ar_b=ar_b,
                      angle_diff=angle_diff)
        done_types.add((type_name, pair_cat))
        saved += 1
        if pair_cat == "floor": saved_floor += 1
        else:                   saved_wall  += 1

    return saved

# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    clean_root  = Path(CLEAN_LAYOUT_ROOT)
    total_saved = 0

    scene_folders = sorted(d.name for d in clean_root.iterdir() if d.is_dir())
    print(f"Found {len(scene_folders)} scenes\n")

    for scene_folder in scene_folders:
        scene_dir = clean_root / scene_folder
        jsons = [f for f in scene_dir.glob("*.json") if f.name not in _SKIP_JSON]
        if not jsons:
            continue

        out_dir  = Path(OUTPUT_BASE) / scene_folder
        existing = list(out_dir.glob("distortion_sizedistance_*/camera_poses.json"))
        if existing:
            print(f"[SKIP] {scene_folder}: already has {len(existing)} variant(s)")
            continue

        print("=" * 60)
        print(f"Scene: {scene_folder}")
        n = generate_sizedistance_for_scene(str(jsons[0]), scene_folder)
        total_saved += n

    print(f"\n{'='*60}")
    print(f"Done. {total_saved} size-distance variant(s) generated.")
