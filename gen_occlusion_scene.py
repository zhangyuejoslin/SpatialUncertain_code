import json
import copy
import math
import random
import os
import numpy as np

# ---------- config ----------
MAX_SHIFT = 5        # maximum lateral shift (meters)
DEPTH_RATIO = (0.7, 0.95)  # occluder is placed closer to target to avoid being near the agent, which makes plausible occlusion difficult
NUM_TRIALS = 50
LATERAL_THRESH = 0.75  # allow larger lateral offset to increase scene generation success rate
# lateral offset configuration per occlusion level (distinguishes 10/50/100 only by distance from center line, depth unchanged)
OCCLUSION_LATERAL_RATIOS = {
    0.1: 0.8,   # 10% occlusion: large lateral offset allowed (farther from center line, less occlusion)
    0.5: 0.4,   # 50% occlusion: moderate lateral offset (near center line)
    1.0: 0.05   # 100% occlusion: very small lateral offset (almost on center line, maximum occlusion)
}  # mapping from occlusion level to lateral offset ratio; smaller value means closer to center line
MIN_DEPTH_SEPARATION = 0.2  # increased minimum front-back separation to avoid penetration
MIN_AABB_SEPARATION = 0.15  # minimum safe distance between occluder and target AABBs (meters)
AGENT_FOOTPRINT_RADIUS = 0.25  # AI2THOR agent radius in the XZ plane (meters)
CAMERA_CLEARANCE_MARGIN = 0.35  # extra safety margin from occluder AABB edge to camera (meters)
CONTACT_TOLERANCE = 0.15  # contact tolerance (meters) for checking whether an object touches the floor/wall
MIN_TARGET_DISTANCE = 1.5  # minimum distance from target to agent (meters) to ensure target is not too close
MAX_TARGET_DISTANCE = 10.0  # maximum distance from target to agent (meters) to avoid targets too far to occlude
OCCLUDER_STOP_WORDS = ["lamp"]  # stop-word list for occluders; objects whose IDs contain these words are excluded (case-insensitive)
ALLOW_LARGER_OCCLUDER = True  # whether to allow selecting an occluder larger than the target (all dimensions)
PREFER_LARGER_OCCLUDER = True  # whether to prefer occluders larger than the target
MIN_OCCLUDER_HEIGHT_RATIO = 0.6  # occluder height must be at least this fraction of target height (e.g. 0.6 means 60%)
ENFORCE_OCCLUDER_HEIGHT = True   # whether to filter out too-short occluders during occluder selection
APPROX_AABB_SCALE = 0.9  # scale factor when approximating size from object_selection_plan to avoid being too conservative
SUBGRAPH_REFERENT_EXCLUSION = False  # occluder can be chosen from anywhere in the scene; referents are not excluded by default
MIN_VERTICAL_OVERLAP_RATIO = 0.15  # allow low vertical overlap to avoid over-rejection when using approximate AABBs
MIN_OCCLUSION_TARGET_DISTANCE = 1.2  # targets too close to the agent leave almost no room for a valid occluder
LOCAL_COLLISION_RADIUS = 0.8  # only resolve collisions with objects near the placement point; ignore distant unrelated objects
APPROX_FOOTPRINT_SCALE = 0.45  # approximate size is only for rough penetration check; use a smaller footprint radius to avoid false positives
MIN_OCCLUDER_FOOTPRINT_RATIO = 0.4  # occluder horizontal footprint must reach at least this fraction of the target's footprint for effective occlusion
MIN_STRONG_OCCLUDER_WIDTH_RATIO = 0.75  # strong occlusion requires the occluder's projected view width to be sufficiently close to the target's
REQUIRE_SAME_AREA = True  # occluder must be in the same area as the target (area name in parentheses of the object id)
# ----------------------------


def vec(p):
    return np.array([p["x"], p["y"], p["z"]], dtype=float)


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-6)


def perpendicular_xz(v):
    perp = np.array([-v[2], 0.0, v[0]], dtype=float)
    if np.linalg.norm(perp) < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return normalize(perp)


def projected_half_width_xz(aabb, perp_dir):
    perp_x, perp_z = abs(perp_dir[0]), abs(perp_dir[2])
    return 0.5 * (perp_x * aabb["size"][0] + perp_z * aabb["size"][2])

def _get_floor_y(scene) -> float:
    """Return the floor Y coordinate of the scene (read from floorPolygon; fallback to 0.0)."""
    rooms = scene.get("rooms", [])
    for room in rooms:
        fp = room.get("floorPolygon")
        if fp and len(fp) > 0 and "y" in fp[0]:
            return float(fp[0]["y"])
    return 0.0


def _read_camera_pos_from_file(camera_poses_path: str) -> np.ndarray | None:
    """Read the camera position of the first frame directly from a camera_poses.json path."""
    try:
        with open(camera_poses_path) as f:
            camera_data = json.load(f)
        seq = (camera_data.get("camera_frame_for_modality_eval")
               or camera_data.get("primary_sequence")
               or "multiview")
        frames = (camera_data.get(seq) or {}).get("frames") or []
        if frames and isinstance(frames[0], dict):
            pos = frames[0].get("position", {})
            return np.array([pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)], dtype=float)
    except Exception as e:
        print(f"[WARN] Failed to load camera_poses.json from {camera_poses_path}: {e}")
    return None


def _load_viewpoint(json_path: str) -> np.ndarray | None:
    """
    Read the primary frame camera position from camera_poses.json in the same directory.
    Returns None if not found; caller should fall back to agent position.
    """
    import pathlib
    camera_poses_path = pathlib.Path(json_path).parent / "camera_poses.json"
    if not camera_poses_path.exists():
        return None
    return _read_camera_pos_from_file(str(camera_poses_path))


def _scene_floor_bounds(scene):
    """
    Return (min_x, max_x, min_z, max_z) from first room's floorPolygon if present.
    Fallback: infer from walls if needed.
    """
    rooms = scene.get("rooms", [])
    if rooms and rooms[0].get("floorPolygon"):
        fp = rooms[0]["floorPolygon"]
        xs = [p["x"] for p in fp]
        zs = [p["z"] for p in fp]
        return min(xs), max(xs), min(zs), max(zs)

    # Fallback: use all wall polygons
    walls = scene.get("walls", [])
    xs, zs = [], []
    for w in walls:
        for p in w.get("polygon", []):
            xs.append(p["x"])
            zs.append(p["z"])
    if xs and zs:
        return min(xs), max(xs), min(zs), max(zs)

    # Last resort
    return -10.0, 10.0, -10.0, 10.0


def _is_wall_object(scene, obj_id: str) -> bool:
    # Prefer explicit lists if present
    wall_objs = scene.get("wall_objects") or scene.get("wallObjects") or []
    for o in wall_objs:
        if o.get("id") == obj_id:
            return True
    return False


def _closest_wall_plane(scene, pos_xyz, margin=0.25):
    """
    Given a world position (x,y,z), determine the closest axis-aligned boundary plane
    for the room (x=min_x/x=max_x or z=min_z/z=max_z).
    Returns (axis, value) where axis in {"x","z"} and value is the plane coordinate.
    """
    x, _, z = pos_xyz
    min_x, max_x, min_z, max_z = _scene_floor_bounds(scene)
    candidates = [
        ("x", min_x, abs(x - min_x)),
        ("x", max_x, abs(x - max_x)),
        ("z", min_z, abs(z - min_z)),
        ("z", max_z, abs(z - max_z)),
    ]
    axis, value, dist = min(candidates, key=lambda t: t[2])
    # If it's not near any wall, still return closest; caller may choose to ignore.
    return axis, value, dist


def _clamp_to_room(scene, pos_xyz, margin=0.05):
    x, y, z = pos_xyz
    min_x, max_x, min_z, max_z = _scene_floor_bounds(scene)
    x = float(np.clip(x, min_x + margin, max_x - margin))
    z = float(np.clip(z, min_z + margin, max_z - margin))
    return np.array([x, y, z], dtype=float)


def _get_object_plan_entry(scene, obj):
    room_id = obj.get("roomId")
    object_selection_plan = scene.get("object_selection_plan", {})
    room_plan = object_selection_plan.get(room_id, {})

    object_name = obj.get("object_name", "") or obj.get("id", "")
    base_name = object_name.rsplit("-", 1)[0]
    return room_plan.get(base_name)


def _get_approx_size_from_plan(scene, obj):
    plan_entry = _get_object_plan_entry(scene, obj)
    if not plan_entry:
        return None

    raw_size = plan_entry.get("size")
    if not isinstance(raw_size, list) or len(raw_size) != 3:
        return None

    # object_selection_plan size is typically [x, z, y] in cm
    size_vec = np.array([
        raw_size[0] / 100.0,
        raw_size[2] / 100.0,
        raw_size[1] / 100.0,
    ], dtype=float)

    rot_y = float(obj.get("rotation", {}).get("y", 0.0)) % 180.0
    if abs(rot_y - 90.0) < 1e-3:
        size_vec[[0, 2]] = size_vec[[2, 0]]

    return size_vec * APPROX_AABB_SCALE


def get_aabb(obj, scene=None, position_override=None):
    """Get the AABB (axis-aligned bounding box) of an object; if missing, approximate from the scene's object plan."""
    aabb = obj.get("axisAlignedBoundingBox", {})
    if not aabb:
        if scene is None:
            return None

        size_vec = _get_approx_size_from_plan(scene, obj)
        if size_vec is None:
            return None

        center_vec = vec(position_override or obj["position"])
        if not _is_wall_object(scene, obj.get("id", "")):
            center_vec = center_vec.copy()
            center_vec[1] = max(center_vec[1], size_vec[1] / 2.0)

        return {
            "center": center_vec,
            "size": size_vec,
            "min": center_vec - size_vec / 2.0,
            "max": center_vec + size_vec / 2.0,
            "source": "approx",
        }
    
    center = aabb.get("center", {})
    size = aabb.get("size", {})
    
    if not all(k in center for k in ["x", "y", "z"]):
        return None
    if not all(k in size for k in ["x", "y", "z"]):
        return None
    
    center_vec = np.array([center["x"], center["y"], center["z"]], dtype=float)
    if position_override is not None:
        center_vec = vec(position_override)
    size_vec = np.array([size["x"], size["y"], size["z"]], dtype=float)
    
    return {
        "center": center_vec,
        "size": size_vec,
        "min": center_vec - size_vec / 2,
        "max": center_vec + size_vec / 2,
        "source": "provided",
    }


def check_front_back_depth(agent, occ, tgt):
    """
    Check front-back depth: ensure the occluder is between the agent and target with sufficient separation.

    Depth is computed as a projection onto the agent->target line of sight, not as a Euclidean distance difference.

    Returns:
        (is_valid, depth_separation): whether valid, and the depth separation distance
    """
    v_t = vec(tgt["position"]) - agent
    v_o = vec(occ["position"]) - agent

    d_t = np.linalg.norm(v_t)
    if d_t < 1e-6:
        return False, 0.0

    # Project onto the agent->target line of sight direction
    dir_t = normalize(v_t)
    depth_t = d_t                        # dot(v_t, dir_t) == d_t
    depth_o = float(np.dot(v_o, dir_t))  # occluder depth along the line of sight

    # occluder must be in front of the agent and in front of the target
    if depth_o <= 0 or depth_o >= depth_t:
        return False, 0.0

    depth_separation = depth_t - depth_o

    if depth_separation < MIN_DEPTH_SEPARATION:
        return False, depth_separation

    return True, depth_separation


def _aabb_axis_gap(min_a, max_a, min_b, max_b):
    if max_a < min_b:
        return min_b - max_a
    if max_b < min_a:
        return min_a - max_b
    return -min(max_a - min_b, max_b - min_a)


def check_no_3d_penetration(scene, occ, tgt):
    """
    Check 3D penetration: ensure the occluder and target AABBs do not intersect and have sufficient clearance.

    Returns:
        (is_valid, min_separation): whether valid (no penetration), and the minimum separation distance
    """
    aabb_occ = get_aabb(occ, scene=scene)
    aabb_tgt = get_aabb(tgt, scene=scene)
    
    # If AABB info is unavailable, fall back to a simple position-based check
    if aabb_occ is None or aabb_tgt is None:
        # Compute distance using positions
        pos_occ = vec(occ["position"])
        pos_tgt = vec(tgt["position"])
        distance = np.linalg.norm(pos_occ - pos_tgt)
        if distance < MIN_AABB_SEPARATION:
            return False, distance
        return True, distance

    if aabb_occ.get("source") == "approx" or aabb_tgt.get("source") == "approx":
        occ_center = aabb_occ["center"]
        tgt_center = aabb_tgt["center"]
        xz_distance = np.linalg.norm((occ_center - tgt_center)[[0, 2]])

        occ_radius = 0.5 * np.linalg.norm(aabb_occ["size"][[0, 2]]) * APPROX_FOOTPRINT_SCALE
        tgt_radius = 0.5 * np.linalg.norm(aabb_tgt["size"][[0, 2]]) * APPROX_FOOTPRINT_SCALE
        required_xz_sep = occ_radius + tgt_radius + MIN_AABB_SEPARATION * 0.5

        vertical_overlap = min(aabb_occ["max"][1], aabb_tgt["max"][1]) - max(aabb_occ["min"][1], aabb_tgt["min"][1])
        if vertical_overlap <= 0:
            return True, xz_distance

        if xz_distance < required_xz_sep:
            return False, xz_distance - required_xz_sep
        return True, xz_distance - required_xz_sep
    
    # Check whether the AABBs intersect
    min_occ, max_occ = aabb_occ["min"], aabb_occ["max"]
    min_tgt, max_tgt = aabb_tgt["min"], aabb_tgt["max"]

    # Compute per-axis gap (positive = separation distance, negative = penetration depth)
    separation_x = _aabb_axis_gap(min_occ[0], max_occ[0], min_tgt[0], max_tgt[0])
    separation_y = _aabb_axis_gap(min_occ[1], max_occ[1], min_tgt[1], max_tgt[1])
    separation_z = _aabb_axis_gap(min_occ[2], max_occ[2], min_tgt[2], max_tgt[2])
    
    # If all axes overlap, the AABBs intersect
    if separation_x < 0 and separation_y < 0 and separation_z < 0:
        # Penetration detected; compute penetration depth
        penetration_depth = max(abs(separation_x), abs(separation_y), abs(separation_z))
        return False, -penetration_depth

    # Compute minimum separation distance across all axes
    min_separation = min(separation_x, separation_y, separation_z)

    # If separation is too small, treat as unsafe (may appear to penetrate visually)
    if min_separation < MIN_AABB_SEPARATION:
        return False, min_separation
    
    return True, min_separation


def check_contact_plausibility(scene, obj):
    """
    Check contact plausibility: ensure the object has reasonable contact with the floor (floor objects) or wall (wall objects).

    Returns:
        (is_valid, contact_info): whether valid, and contact info
    """
    obj_id = obj.get("id", "")
    obj_pos = vec(obj["position"])
    is_wall = _is_wall_object(scene, obj_id)
    
    if is_wall:
        # For wall objects, check proximity to the wall surface
        axis, plane_value, dist = _closest_wall_plane(scene, obj_pos, margin=0.25)

        # Check whether the object is on the wall (taking AABB size into account)
        aabb = get_aabb(obj, scene=scene)
        if aabb is not None:
            # Check the distance from the AABB boundary to the wall plane based on wall orientation
            if axis == "x":
                # Wall at x=plane_value; check AABB x boundaries
                aabb_min_x = aabb["min"][0]
                aabb_max_x = aabb["max"][0]
                # Object should be flush with the wall, so the AABB boundary should be near the wall plane
                min_distance = min(abs(aabb_min_x - plane_value), abs(aabb_max_x - plane_value))
            else:  # axis == "z"
                # Wall at z=plane_value; check AABB z boundaries
                aabb_min_z = aabb["min"][2]
                aabb_max_z = aabb["max"][2]
                min_distance = min(abs(aabb_min_z - plane_value), abs(aabb_max_z - plane_value))

            # Object must be very close to the wall (within tolerance)
            if min_distance > CONTACT_TOLERANCE:
                return False, {"type": "wall", "distance": min_distance, "axis": axis}

            # Check that Y coordinate is reasonable (should not be below ground)
            if aabb["min"][1] < -0.1:
                return False, {"type": "wall", "error": "below_ground"}
        else:
            # No AABB available; use position for the check
            if axis == "x":
                wall_distance = abs(obj_pos[0] - plane_value)
            else:
                wall_distance = abs(obj_pos[2] - plane_value)

            if wall_distance > CONTACT_TOLERANCE:
                return False, {"type": "wall", "distance": wall_distance, "axis": axis}

            if obj_pos[1] < -0.1:
                return False, {"type": "wall", "error": "below_ground"}
        
        return True, {"type": "wall", "axis": axis, "distance": dist}
    else:
        # For floor objects, check whether the object rests on the floor
        floor_y = _get_floor_y(scene)

        # Check whether the object's bottom is close to the floor
        aabb = get_aabb(obj, scene=scene)
        if aabb is not None:
            obj_bottom = aabb["min"][1]
            ground_distance = abs(obj_bottom - floor_y)

            if ground_distance > CONTACT_TOLERANCE:
                return False, {"type": "floor", "distance": ground_distance}
        else:
            # No AABB available; use the y coordinate of the position
            if obj_pos[1] < -CONTACT_TOLERANCE or obj_pos[1] > 2.0:  # abnormal height
                return False, {"type": "floor", "error": "invalid_height", "y": obj_pos[1]}
        
        return True, {"type": "floor"}


def occludes(scene, agent, occ, tgt):
    """
    Check whether the occluder occludes the target.
    Kept permissive: only checks "in front of target and close to the line of sight";
    finer geometry/contact constraints are delegated to validate_occlusion.
    """
    v_t = vec(tgt["position"]) - agent
    v_o = vec(occ["position"]) - agent

    d_t = np.linalg.norm(v_t)
    d_o = np.linalg.norm(v_o)

    # occluder must be closer than target
    if d_o >= d_t:
        return False

    dir_t = normalize(v_t)
    perp_t = perpendicular_xz(dir_t)
    occ_proj = np.dot(v_o, dir_t)
    if occ_proj <= 0:
        return False

    # Compute lateral offset of the occluder relative to the agent->target line of sight
    implies = v_o - occ_proj * dir_t
    lateral = np.linalg.norm(implies)

    # Dynamically relax lateral tolerance based on occluder / target sizes
    aabb_occ = get_aabb(occ, scene=scene)
    aabb_tgt = get_aabb(tgt, scene=scene)
    if aabb_occ is not None and aabb_tgt is not None:
        occ_half_width = projected_half_width_xz(aabb_occ, perp_t)
        tgt_half_width = projected_half_width_xz(aabb_tgt, perp_t)
        occ_width = occ_half_width * 2.0
        tgt_width = tgt_half_width * 2.0

        if tgt_width > 1e-6 and occ_width < MIN_STRONG_OCCLUDER_WIDTH_RATIO * tgt_width:
            return False

        # strong occlusion: occluder center must be close to the target line of sight,
        # and the occluder's projected width must be wide enough to cover the target width
        lateral_limit = max(0.15, occ_half_width - 0.25 * tgt_half_width)
        lateral_limit = min(lateral_limit, 0.6 * tgt_half_width + 0.1)

        # Keep a very loose vertical compatibility check to avoid obviously impossible cases
        min_occ_y, max_occ_y = aabb_occ["min"][1], aabb_occ["max"][1]
        min_tgt_y, max_tgt_y = aabb_tgt["min"][1], aabb_tgt["max"][1]
        vertical_overlap = min(max_occ_y, max_tgt_y) - max(min_occ_y, min_tgt_y)
        if vertical_overlap < -0.15:
            return False
    else:
        lateral_limit = LATERAL_THRESH

    return lateral <= lateral_limit


def check_camera_clearance(agent, occ, scene):
    """
    Ensure the occluder AABB does not overlap the camera (agent) position.
    Prevents placing the occluder in the same grid cell as the camera, which would cause TeleportFull to fail.

    Returns:
        (is_valid, clearance): positive clearance means safe distance (meters), negative means penetration depth
    """
    min_dist = AGENT_FOOTPRINT_RADIUS + CAMERA_CLEARANCE_MARGIN
    cam_x, cam_z = float(agent[0]), float(agent[2])

    aabb = get_aabb(occ, scene=scene)
    if aabb is None:
        occ_pos = np.array([occ["position"]["x"], occ["position"]["z"]])
        dist = np.linalg.norm(occ_pos - np.array([cam_x, cam_z]))
        return dist >= min_dist, dist - min_dist

    if aabb.get("source") == "approx":
        center = aabb["center"]
        radius = 0.5 * np.linalg.norm(aabb["size"][[0, 2]])
        dist = np.linalg.norm(np.array([center[0], center[2]]) - np.array([cam_x, cam_z]))
        clearance = dist - radius - min_dist
        return clearance >= 0, clearance

    # Precise AABB: compute shortest distance from camera point to occluder XZ rectangle
    mn, mx = aabb["min"], aabb["max"]
    dx = max(mn[0] - cam_x, 0.0, cam_x - mx[0])
    dz = max(mn[2] - cam_z, 0.0, cam_z - mx[2])
    dist_to_edge = float(np.sqrt(dx * dx + dz * dz))
    clearance = dist_to_edge - min_dist
    return clearance >= 0, clearance


def validate_occlusion(scene, agent, occ, tgt):
    """
    Validate whether the occlusion scene satisfies all constraints.

    Returns:
        (is_valid, validation_info): whether valid, and validation info
    """
    validation_info = {
        "front_back_depth": None,
        "no_penetration": None,
        "contact_plausibility_occ": None,
        "contact_plausibility_tgt": None,
        "camera_clearance": None,
    }
    
    # 1. Check front-back depth
    depth_valid, depth_sep = check_front_back_depth(agent, occ, tgt)
    validation_info["front_back_depth"] = {
        "valid": depth_valid,
        "depth_separation": depth_sep
    }
    if not depth_valid:
        return False, validation_info
    
    # 2. Check 3D penetration
    penetration_valid, penetration_depth = check_no_3d_penetration(scene, occ, tgt)
    validation_info["no_penetration"] = {
        "valid": penetration_valid,
        "penetration_depth": penetration_depth
    }
    if not penetration_valid:
        return False, validation_info
    
    # 3. Check contact plausibility (occluder)
    contact_occ_valid, contact_occ_info = check_contact_plausibility(scene, occ)
    validation_info["contact_plausibility_occ"] = {
        "valid": contact_occ_valid,
        "info": contact_occ_info
    }
    if not contact_occ_valid:
        return False, validation_info
    
    # 4. Check contact plausibility (target; optional but recommended)
    contact_tgt_valid, contact_tgt_info = check_contact_plausibility(scene, tgt)
    validation_info["contact_plausibility_tgt"] = {
        "valid": contact_tgt_valid,
        "info": contact_tgt_info
    }
    # Note: a failed target contact check does not necessarily invalidate the whole scene, but it is recorded

    # 5. Check that occluder does not cover the camera position (prevents TeleportFull failure)
    camera_clear, clearance = check_camera_clearance(agent, occ, scene)
    validation_info["camera_clearance"] = {
        "valid": camera_clear,
        "clearance_m": round(clearance, 3),
    }
    if not camera_clear:
        return False, validation_info

    return True, validation_info


def verify_scene_isolation(original_scene, new_scene, occluder_id):
    """
    Verify scene isolation: ensure only the occluder's position was modified; all other objects remain unchanged.

    Returns:
        (is_isolated, differences): whether isolated, and a list of differences
    """
    differences = []
    
    # Build a mapping from object ID to object
    orig_objects = {o["id"]: o for o in original_scene.get("objects", [])}
    new_objects = {o["id"]: o for o in new_scene.get("objects", [])}
    
    for obj_id in orig_objects:
        if obj_id not in new_objects:
            differences.append(f"Object {obj_id} missing in new scene")
            continue
        
        orig_obj = orig_objects[obj_id]
        new_obj = new_objects[obj_id]
        
        # For the occluder, the position should differ
        if obj_id == occluder_id:
            orig_pos = vec(orig_obj["position"])
            new_pos = vec(new_obj["position"])
            if np.allclose(orig_pos, new_pos):
                differences.append(f"Occluder {obj_id} position unchanged!")
        else:
            # For all other objects, the position should be unchanged
            orig_pos = vec(orig_obj["position"])
            new_pos = vec(new_obj["position"])
            if not np.allclose(orig_pos, new_pos, atol=1e-6):
                differences.append(f"Non-occluder {obj_id} position changed: {orig_pos} -> {new_pos}")
    
    return len(differences) == 0, differences


def _get_subgraph_referent_ids(target_subgraphs, target_id):
    if not target_subgraphs:
        return set()

    target_entry = target_subgraphs.get(target_id, {})
    relations = target_entry.get("relations", {})
    referent_ids = set()

    for pairs in relations.values():
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            if pair[0] == target_id and isinstance(pair[1], str):
                referent_ids.add(pair[1])

    return referent_ids


def _get_local_collision_objects(scene, occ_id, target_id, desired_pos):
    local_objects = []
    desired = np.array(desired_pos, dtype=float)

    for other in scene.get("objects", []):
        other_id = other.get("id")
        if other_id == occ_id:
            continue
        if other_id == target_id:
            local_objects.append(other)
            continue

        other_pos = vec(other["position"])
        xz_dist = np.linalg.norm((other_pos - desired)[[0, 2]])
        if xz_dist <= LOCAL_COLLISION_RADIUS:
            local_objects.append(other)

    return local_objects


def _resolve_occluder_penetration(scene, occ, target_id, desired_pos, locked_axis=None):
    """
    Push the occluder to a position that does not penetrate any other object.

    Returns:
        (resolved_pos, success)
    """
    resolved = desired_pos.copy()
    occ_id = occ.get("id")
    occ_aabb = get_aabb(occ, scene=scene, position_override={"x": resolved[0], "y": resolved[1], "z": resolved[2]})
    if occ_aabb is None:
        return resolved, True

    for _ in range(12):
        collision_found = False

        for other in _get_local_collision_objects(scene, occ_id, target_id, resolved):

            other_aabb = get_aabb(other, scene=scene)
            occ_aabb = get_aabb(occ, scene=scene, position_override={"x": resolved[0], "y": resolved[1], "z": resolved[2]})
            if other_aabb is None or occ_aabb is None:
                continue

            gap_x = _aabb_axis_gap(occ_aabb["min"][0], occ_aabb["max"][0], other_aabb["min"][0], other_aabb["max"][0])
            gap_y = _aabb_axis_gap(occ_aabb["min"][1], occ_aabb["max"][1], other_aabb["min"][1], other_aabb["max"][1])
            gap_z = _aabb_axis_gap(occ_aabb["min"][2], occ_aabb["max"][2], other_aabb["min"][2], other_aabb["max"][2])

            if gap_x >= MIN_AABB_SEPARATION or gap_y >= MIN_AABB_SEPARATION or gap_z >= MIN_AABB_SEPARATION:
                continue

            collision_found = True
            push = np.zeros(3, dtype=float)
            other_center = other_aabb["center"]
            delta = occ_aabb["center"] - other_center

            if locked_axis == "x":
                direction = 1.0 if delta[2] >= 0 else -1.0
                push[2] = direction * (MIN_AABB_SEPARATION - gap_z + 1e-3)
            elif locked_axis == "z":
                direction = 1.0 if delta[0] >= 0 else -1.0
                push[0] = direction * (MIN_AABB_SEPARATION - gap_x + 1e-3)
            else:
                horizontal_delta = np.array([delta[0], 0.0, delta[2]], dtype=float)
                if np.linalg.norm(horizontal_delta) < 1e-6:
                    horizontal_delta = np.array([1.0, 0.0, 0.0], dtype=float)
                horizontal_dir = normalize(horizontal_delta)
                required_push = max(MIN_AABB_SEPARATION - min(gap_x, gap_z), MIN_AABB_SEPARATION)
                push = horizontal_dir * (required_push + 1e-3)
                push[1] = 0.0

            resolved = _clamp_to_room(scene, resolved + push, margin=0.05)
            break

        if not collision_found:
            return resolved, True

    return resolved, False


def _update_object_position_and_aabb(scene, obj, new_pos):
    obj["position"]["x"] = float(new_pos[0])
    obj["position"]["y"] = float(new_pos[1])
    obj["position"]["z"] = float(new_pos[2])

    obj_aabb = get_aabb(obj, scene=scene, position_override=obj["position"])
    if obj_aabb is not None:
        obj["axisAlignedBoundingBox"] = {
            "center": {
                "x": float(obj_aabb["center"][0]),
                "y": float(obj_aabb["center"][1]),
                "z": float(obj_aabb["center"][2]),
            },
            "size": {
                "x": float(obj_aabb["size"][0]),
                "y": float(obj_aabb["size"][1]),
                "z": float(obj_aabb["size"][2]),
            }
        }


def move_occluder(scene, agent, target, occ, occlusion_ratio=None):
    """
    Move occluder towards the agent->target ray to occlude the target, while preserving
    its support constraints:
    - Wall objects: keep on the same wall plane (lock wall-normal axis) and keep original y.
    - Floor objects: keep original y (approximately on floor) and move in XZ plane.
    
    Args:
        occlusion_ratio: occlusion level (0.0-1.0); None means random
            - 0.3 (30%): occluder slightly off the agent->target center line, less occlusion
            - 0.6 (60%): occluder closer to the center line, moderate occlusion
            - 1.0 (100%): occluder fully on the agent->target center line, maximum occlusion
    
    NOTE: This function ONLY modifies the position of the occluder object (occ).
    All other objects in the scene remain unchanged.
    
    IMPORTANT: The occ object should be from a fresh copy of the original scene,
    so occ_pos0 will be the original position.
    """
    v = vec(target["position"]) - agent
    d = float(np.linalg.norm(v))
    if d < 1e-6:
        return False
    dir_v = normalize(v)
    perp_v = perpendicular_xz(dir_v)

    occ_id = occ.get("id", "")
    # Get the occluder's current position (should be the original position since it was copied from the original scene)
    occ_pos0 = vec(occ["position"])

    # Get occluder and target AABB sizes to adjust the lateral offset
    aabb_occ = get_aabb(occ, scene=scene)
    aabb_tgt = get_aabb(target, scene=scene)

    # Compute a size-based adjustment factor.
    # A larger occluder can tolerate a larger lateral offset (it can still occlude even when more off-center).
    # A smaller occluder needs a smaller lateral offset (must stay closer to the center line to occlude).
    if aabb_occ is not None and aabb_tgt is not None:
        # Use the maximum dimension of occluder and target
        occ_max_size = np.max(aabb_occ["size"])
        tgt_max_size = np.max(aabb_tgt["size"])

        # Size ratio: if occluder is larger than target, allow a larger lateral offset.
        # size_factor == 1.0 means same size; > 1.0 means occluder is larger.
        size_factor = occ_max_size / (tgt_max_size + 1e-6)

        # Adjustment factor: larger occluder allows larger lateral offset (capped at 2x).
        # For very small occluders, lateral offset is more tightly controlled.
        size_adjustment = min(2.0, max(0.5, size_factor))
    else:
        # No AABB info available; use default
        size_adjustment = 1.0

    is_wall = _is_wall_object(scene, occ_id)

    if is_wall:
        # Lock to closest wall plane near original position
        axis, plane_value, dist = _closest_wall_plane(scene, occ_pos0, margin=0.25)
        # If it's really far from any wall, fall back to unconstrained
        if dist > 0.6:
            is_wall = False

    if occlusion_ratio is not None:
        lateral_ratio = OCCLUSION_LATERAL_RATIOS.get(occlusion_ratio, 0.5)
        if occlusion_ratio < 0.5:
            lateral_adjustment = size_adjustment
        else:
            lateral_adjustment = 1.0 / size_adjustment
        max_lateral_shift = MAX_SHIFT * lateral_ratio * lateral_adjustment
    else:
        max_lateral_shift = MAX_SHIFT * size_adjustment

    depth_ratios = [DEPTH_RATIO[1], 0.9, 0.82, 0.74, DEPTH_RATIO[0]]
    random_depths = [random.uniform(DEPTH_RATIO[0], DEPTH_RATIO[1]) for _ in range(4)]
    depth_ratios.extend(random_depths)
    shift_factors = [0.0, -0.15, 0.15, -0.3, 0.3, -0.5, 0.5, -0.75, 0.75]
    shift_factors.extend([random.uniform(-1.0, 1.0) for _ in range(6)])

    original_position = dict(occ["position"])
    original_aabb = copy.deepcopy(occ.get("axisAlignedBoundingBox"))
    best_candidate = None
    best_score = float("inf")
    best_candidate_fully_valid = False  # True only if penetration_valid AND occludes_valid

    for depth_ratio in depth_ratios:
        depth = depth_ratio * d
        base = agent + depth * dir_v

        for shift_factor in shift_factors:
            shift = max_lateral_shift * shift_factor
            new_pos = base.copy()

            if is_wall:
                if axis == "x":
                    new_pos[0] = plane_value
                    new_pos[2] = new_pos[2] + shift
                else:
                    new_pos[2] = plane_value
                    new_pos[0] = new_pos[0] + shift
                new_pos[1] = occ_pos0[1]
                new_pos = _clamp_to_room(scene, new_pos, margin=0.05)
                new_pos, resolved_ok = _resolve_occluder_penetration(scene, occ, target["id"], new_pos, locked_axis=axis)
            else:
                new_pos = base + perp_v * shift
                new_pos[1] = occ_pos0[1]
                new_pos = _clamp_to_room(scene, new_pos, margin=0.05)
                new_pos, resolved_ok = _resolve_occluder_penetration(scene, occ, target["id"], new_pos, locked_axis=None)

            if not resolved_ok:
                continue

            _update_object_position_and_aabb(scene, occ, new_pos)
            v_o = vec(occ["position"]) - agent
            occ_proj = np.dot(v_o, dir_v)
            lateral = np.linalg.norm(v_o - occ_proj * dir_v)
            penetration_valid, penetration_depth = check_no_3d_penetration(scene, occ, target)
            occludes_valid = occludes(scene, agent, occ, target)

            score = lateral
            if occ_proj >= d:
                score += 10.0 + (occ_proj - d)
            if not penetration_valid:
                score += 5.0 + abs(penetration_depth)
            if not occludes_valid:
                score += 2.0
            score += abs(1.0 - depth_ratio) * 0.5

            if penetration_valid and occludes_valid:
                return True

            is_fully_valid = penetration_valid and occludes_valid
            if score < best_score or (is_fully_valid and not best_candidate_fully_valid):
                best_score = score
                best_candidate = {
                    "position": dict(occ["position"]),
                    "aabb": copy.deepcopy(occ.get("axisAlignedBoundingBox")),
                }
                best_candidate_fully_valid = is_fully_valid

    if best_candidate is not None and best_candidate_fully_valid:
        occ["position"] = best_candidate["position"]
        if best_candidate["aabb"] is None:
            occ.pop("axisAlignedBoundingBox", None)
        else:
            occ["axisAlignedBoundingBox"] = best_candidate["aabb"]
        return True

    occ["position"] = original_position
    if original_aabb is None:
        occ.pop("axisAlignedBoundingBox", None)
    else:
        occ["axisAlignedBoundingBox"] = original_aabb
    return False


def generate_for_objects(json_path, occluder_id, target_id, num_scenes=10):
    """
    Generate a series of occlusion scenes for the specified pair of objects.

    Args:
        json_path: path to the scene JSON file
        occluder_id: ID of the occluder object
        target_id: ID of the target object
        num_scenes: number of scenes to generate

    Returns:
        list of generated occlusion scenes
    """
    with open(json_path) as f:
        original_scene = json.load(f)  # save original scene to always start from a clean state

    agent = vec(original_scene["metadata"]["agent"]["position"])
    objects = original_scene["objects"]

    # Find the specified objects (looked up in the original scene for validation)
    occluder = next((o for o in objects if o["id"] == occluder_id), None)
    target = next((o for o in objects if o["id"] == target_id), None)
    
    if occluder is None:
        raise ValueError(f"Occluder object '{occluder_id}' not found in scene")
    if target is None:
        raise ValueError(f"Target object '{target_id}' not found in scene")
    
    if occluder_id == target_id:
        raise ValueError("Occluder and target cannot be the same object")

    results = []
    attempts = 0
    max_attempts = NUM_TRIALS * num_scenes * 2  # increased attempt limit

    while len(results) < num_scenes and attempts < max_attempts:
        attempts += 1
        # Create a fresh deep copy of the original scene each iteration to ensure full independence.
        # Important: use original_scene (not scene) so every scene starts from the original state.
        new_scene = copy.deepcopy(original_scene)
        occ_new = next(o for o in new_scene["objects"] if o["id"] == occluder_id)
        tgt_new = next(o for o in new_scene["objects"] if o["id"] == target_id)

        # Use new_scene (not the original scene) to ensure only the current scene is modified
        moved = move_occluder(new_scene, agent, tgt_new, occ_new)
        if not moved:
            continue

        # Basic occlusion check
        if not occludes(new_scene, agent, occ_new, tgt_new):
            continue

        # Detailed constraint validation (using new_scene)
        is_valid, validation_info = validate_occlusion(new_scene, agent, occ_new, tgt_new)
        if not is_valid:
            # Optional: print validation failure reason (for debugging)
            if attempts % 10 == 0:  # print once every 10 attempts
                print(f"  [DEBUG] Validation failed: depth={validation_info['front_back_depth']['valid']}, "
                      f"penetration={validation_info['no_penetration']['valid']}, "
                      f"contact_occ={validation_info['contact_plausibility_occ']['valid']}")
            continue
        
        # All checks passed; save the scene.
        # Ensure full independence: only the occluder's position was modified.
        new_scene["occlusion"] = {
            "occluder": occluder_id,
            "target": target_id,
            "validation": validation_info  # save validation info for debugging
        }
        results.append(new_scene)
        print(f"Generated occlusion scene {len(results)}/{num_scenes} (occluder: {occluder_id}, target: {target_id})")

    return results


def _is_occluder_larger_than_target(scene, occ_obj, tgt_obj) -> bool:
    """
    Check whether the occluder is larger than the target (all dimensions).

    Args:
        occ_obj: occluder object
        tgt_obj: target object

    Returns:
        True if the occluder is larger than the target in all dimensions, False otherwise
    """
    aabb_occ = get_aabb(occ_obj, scene=scene)
    aabb_tgt = get_aabb(tgt_obj, scene=scene)
    
    if aabb_occ is None or aabb_tgt is None:
        return False  # Cannot compare without AABB info

    size_occ = aabb_occ["size"]
    size_tgt = aabb_tgt["size"]

    # Use volume comparison: an occluder with greater volume is considered "larger".
    # Requiring all three dimensions to be larger was too strict (e.g. a tall cabinet occluding a short wide object would be incorrectly excluded).
    vol_occ = float(size_occ[0]) * float(size_occ[1]) * float(size_occ[2])
    vol_tgt = float(size_tgt[0]) * float(size_tgt[1]) * float(size_tgt[2])
    return vol_occ > vol_tgt


def _is_occluder_tall_enough(scene, occ_obj, tgt_obj, min_ratio: float) -> bool:
    """
    Check whether the occluder is tall enough relative to the target.

    Args:
        occ_obj: occluder object
        tgt_obj: target object
        min_ratio: minimum height ratio (e.g. 0.6 means occluder height must be at least 60% of target height)

    Returns:
        True if occluder height >= min_ratio * target height, False otherwise
    """
    aabb_occ = get_aabb(occ_obj, scene=scene)
    aabb_tgt = get_aabb(tgt_obj, scene=scene)
    
    if aabb_occ is None or aabb_tgt is None:
        # No AABB info; do not enforce height constraint
        return True
    
    occ_height = aabb_occ["size"][1]
    tgt_height = aabb_tgt["size"][1]
    
    if tgt_height <= 1e-6:
        return True
    
    return occ_height >= min_ratio * tgt_height


def _is_occluder_wide_enough(scene, occ_obj, tgt_obj, min_ratio: float) -> bool:
    aabb_occ = get_aabb(occ_obj, scene=scene)
    aabb_tgt = get_aabb(tgt_obj, scene=scene)

    if aabb_occ is None or aabb_tgt is None:
        return True

    occ_footprint = np.linalg.norm(aabb_occ["size"][[0, 2]])
    tgt_footprint = np.linalg.norm(aabb_tgt["size"][[0, 2]])

    if tgt_footprint <= 1e-6:
        return True

    return occ_footprint >= min_ratio * tgt_footprint


def _get_object_area(obj_id: str) -> str | None:
    """Extract area name from object id, e.g. 'bench-0 (reading nook)' → 'reading nook'."""
    import re
    m = re.search(r'\(([^)]+)\)$', obj_id.strip())
    return m.group(1).strip().lower() if m else None


def _get_object_category(obj_id: str) -> str:
    """Extract category from object id, e.g. 'rolling_laundry_cart-3 (laundry area)' → 'rolling_laundry_cart'."""
    import re
    m = re.match(r'^(.+)-\d+', obj_id.strip())
    return m.group(1) if m else obj_id


def _point_in_polygon_xz(px: float, pz: float, polygon: list) -> bool:
    """Ray casting point-in-polygon test in the XZ plane."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, zi = polygon[i]["x"], polygon[i]["z"]
        xj, zj = polygon[j]["x"], polygon[j]["z"]
        if ((zi > pz) != (zj > pz)) and (px < (xj - xi) * (pz - zi) / (zj - zi + 1e-10) + xi):
            inside = not inside
        j = i
    return inside


def _get_camera_area(scene: dict, camera_pos) -> str | None:
    """
    Determine the area name of the camera by:
    1. Point-in-polygon test against each room's floorPolygon to find the camera's room.
    2. Look up any object in that room and extract its area name from the object id.
    Returns the area name (lowercase) or None if undeterminable.
    """
    cam_x, cam_z = float(camera_pos[0]), float(camera_pos[2])

    camera_room_id = None
    for room in scene.get("rooms", []):
        fp = room.get("floorPolygon")
        if not fp:
            continue
        if _point_in_polygon_xz(cam_x, cam_z, fp):
            camera_room_id = room.get("id") or room.get("roomId")
            break

    if camera_room_id is None:
        return None

    for obj in scene.get("objects", []):
        if obj.get("roomId") == camera_room_id:
            area = _get_object_area(obj["id"])
            if area is not None:
                return area

    return None


def _is_occluder_allowed(obj_id: str, stop_words: list) -> bool:
    """
    Check whether an object is allowed as an occluder (not in the stop-words list).

    Matching rule: if the object ID contains any stop word (case-insensitive), it is excluded.
    Example: "floor_lamp-0 (bedroom)" contains "lamp" and will be excluded.

    Args:
        obj_id: object ID (e.g. "floor_lamp-0 (living room)", "table_lamp-1 (bedroom)")
        stop_words: list of stop words (e.g. ["lamp"])

    Returns:
        True if the object can be used as an occluder, False if excluded by stop words
    """
    if not stop_words:
        return True
    
    obj_id_lower = obj_id.lower()
    for stop_word in stop_words:
        stop_word_lower = stop_word.lower()
        # Check whether the stop word appears in the object ID (substring match)
        if stop_word_lower in obj_id_lower:
            return False
    return True


def load_target_subgraph_data(target_subgraph_path):
    if not target_subgraph_path:
        return {}

    if not os.path.exists(target_subgraph_path):
        print(f"[WARN] target subgraph file not found: {target_subgraph_path}")
        return {}

    try:
        with open(target_subgraph_path) as f:
            subgraph_data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load target subgraph file {target_subgraph_path}: {e}")
        return {}

    if not isinstance(subgraph_data, dict):
        print(f"[WARN] target subgraph content is not a dict in {target_subgraph_path}")
        return {}

    return subgraph_data


def load_target_ids_from_subgraph(target_subgraph_path):
    """
    Read target_ids from target_subgraph.json.

    Returns:
        list[str]: list of target IDs; returns an empty list if the file is missing or malformed.
    """
    subgraph_data = load_target_subgraph_data(target_subgraph_path)
    if not subgraph_data:
        return []

    target_ids = subgraph_data.get("target_ids", [])
    if not isinstance(target_ids, list):
        print(f"[WARN] target_ids is not a list in {target_subgraph_path}")
        return []

    filtered_target_ids = [target_id for target_id in target_ids if isinstance(target_id, str)]
    if len(filtered_target_ids) != len(target_ids):
        print(f"[WARN] Ignored non-string target ids in {target_subgraph_path}")

    return filtered_target_ids


def get_matching_target_ids_for_scene(scene, target_ids):
    if not target_ids:
        return []

    scene_object_ids = {obj["id"] for obj in scene.get("objects", [])}
    return [target_id for target_id in target_ids if target_id in scene_object_ids]


def generate_by_target(
    json_path,
    num_occluders_per_target=10,
    target_ids=None,
    target_subgraphs=None,
    rendered_scene_root=None,
):
    """
    Generate occlusion scenes grouped by target.
    - Each target stays in place.
    - Multiple occluder objects are randomly selected to occlude the target (objects matching stop words are excluded).
    - Targets farther from the agent are preferred (distance between MIN_TARGET_DISTANCE and MAX_TARGET_DISTANCE).
    - If target_ids is provided, only those targets are used.
    - Returns a dict of scenes grouped by target: {target_id: [scenes]}
    """
    import pathlib
    with open(json_path) as f:
        original_scene = json.load(f)  # save original scene to always start from a clean state

    # Prefer camera_poses.json under rendered_scene_root (consistent with the rendered viewpoint)
    camera_pos = None
    if rendered_scene_root:
        scene_id = pathlib.Path(json_path).parent.name
        ref_poses = pathlib.Path(rendered_scene_root) / scene_id / "camera_poses.json"
        if ref_poses.exists():
            camera_pos = _read_camera_pos_from_file(str(ref_poses))
            if camera_pos is not None:
                print(f"[INFO] Using camera position from rendered_scene_root: {camera_pos}")
    if camera_pos is None:
        camera_pos = _load_viewpoint(json_path)
    if camera_pos is not None:
        agent = camera_pos
        print(f"[INFO] Using camera position from camera_poses.json: {agent}")
    else:
        agent = vec(original_scene["metadata"]["agent"]["position"])
        print(f"[INFO] camera_poses.json not found, using agent position: {agent}")
    objects = original_scene["objects"]
    object_by_id = {obj["id"]: obj for obj in objects}

    # Determine the camera's area; only generate occlusion for targets in the same area
    camera_area = _get_camera_area(original_scene, agent)
    if camera_area is not None:
        print(f"[INFO] Camera area: '{camera_area}' — only targets in this area will be used")
    else:
        print("[WARN] Could not determine camera area; targets not filtered by area")

    # Results grouped by target
    results_by_target = {}

    # Compute each object's distance to the agent and filter valid targets
    valid_targets = []
    requested_target_ids = get_matching_target_ids_for_scene(original_scene, target_ids or [])

    if requested_target_ids:
        for target_id in requested_target_ids:
            target = object_by_id.get(target_id)
            if target is None:
                continue
            target_distance = np.linalg.norm(vec(target["position"]) - agent)
            valid_targets.append((target, target_distance))

        valid_targets.sort(key=lambda x: x[1], reverse=True)
        print(f"[INFO] Using {len(valid_targets)} targets from target_subgraph")
    else:
        for obj in objects:
            obj_pos = vec(obj["position"])
            distance = np.linalg.norm(obj_pos - agent)

            # Only select objects within a reasonable distance range as targets
            if MIN_TARGET_DISTANCE <= distance <= MAX_TARGET_DISTANCE:
                valid_targets.append((obj, distance))

        # Sort by distance descending (prefer farther objects)
        valid_targets.sort(key=lambda x: x[1], reverse=True)

        if len(valid_targets) == 0:
            print(f"[WARN] No valid targets found (distance range: {MIN_TARGET_DISTANCE}-{MAX_TARGET_DISTANCE}m)")
            print(f"[INFO] Available objects distances: {[np.linalg.norm(vec(o['position']) - agent) for o in objects]}")
            # No suitable targets; fall back to all objects (no distance restriction)
            valid_targets = [(obj, np.linalg.norm(vec(obj["position"]) - agent)) for obj in objects]
            valid_targets.sort(key=lambda x: x[1], reverse=True)

        print(f"[INFO] Found {len(valid_targets)} valid targets (sorted by distance, farthest first)")

    # Keep only targets in the same area as the camera
    if camera_area is not None:
        before = len(valid_targets)
        valid_targets = [
            (t, d) for t, d in valid_targets
            if _get_object_area(t["id"]) == camera_area
        ]
        print(f"[INFO] Area filter '{camera_area}': {before} → {len(valid_targets)} targets")

    # For each target, randomly select multiple occluders
    for target, target_distance in valid_targets:
        target_id = target["id"]
        results_by_target[target_id] = []

        if target_distance < MIN_OCCLUSION_TARGET_DISTANCE:
            print(f"[INFO] Skip target {target_id}: too close to agent for stable occlusion ({target_distance:.2f}m)")
            continue
        
        referent_ids = _get_subgraph_referent_ids(target_subgraphs, target_id)

        # Gather all candidate occluders (exclude the target itself, same-category objects, and stop-word objects)
        target_category = _get_object_category(target_id)
        possible_occluders = [
            obj for obj in objects
            if obj["id"] != target_id
            and _get_object_category(obj["id"]) != target_category
            and _is_occluder_allowed(obj["id"], OCCLUDER_STOP_WORDS)
        ]

        if SUBGRAPH_REFERENT_EXCLUSION and referent_ids:
            possible_occluders = [
                obj for obj in possible_occluders
                if obj["id"] not in referent_ids
            ]
        
        # If larger occluders are not allowed, filter out objects larger than the target
        if not ALLOW_LARGER_OCCLUDER:
            possible_occluders = [
                obj for obj in possible_occluders
                if not _is_occluder_larger_than_target(original_scene, obj, target)
            ]

        # Optional: filter out obviously too-short occluders at selection time
        if ENFORCE_OCCLUDER_HEIGHT:
            possible_occluders = [
                obj for obj in possible_occluders
                if _is_occluder_tall_enough(original_scene, obj, target, MIN_OCCLUDER_HEIGHT_RATIO)
            ]

        possible_occluders = [
            obj for obj in possible_occluders
            if _is_occluder_wide_enough(original_scene, obj, target, MIN_OCCLUDER_FOOTPRINT_RATIO)
        ]

        if REQUIRE_SAME_AREA:
            target_area = _get_object_area(target_id)
            if target_area is not None:
                possible_occluders = [
                    obj for obj in possible_occluders
                    if _get_object_area(obj["id"]) == target_area
                ]

        # Count filtered-out objects (for debugging)
        filtered_out_stopwords = [
            obj["id"] for obj in objects 
            if obj["id"] != target_id 
            and not _is_occluder_allowed(obj["id"], OCCLUDER_STOP_WORDS)
        ]
        if filtered_out_stopwords:
            print(f"[INFO] Filtered out {len(filtered_out_stopwords)} occluders (stop words: {OCCLUDER_STOP_WORDS}): {filtered_out_stopwords}")

        if SUBGRAPH_REFERENT_EXCLUSION and referent_ids:
            print(f"[INFO] Filtered out {len(referent_ids)} referent occluders from subgraph: {sorted(referent_ids)}")
        
        if not ALLOW_LARGER_OCCLUDER:
            filtered_out_larger = [
                obj["id"] for obj in objects 
                if obj["id"] != target_id 
                and _is_occluder_allowed(obj["id"], OCCLUDER_STOP_WORDS)
                and _is_occluder_larger_than_target(original_scene, obj, target)
            ]
            if filtered_out_larger:
                print(f"[INFO] Filtered out {len(filtered_out_larger)} occluders (larger than target): {filtered_out_larger}")

        if ENFORCE_OCCLUDER_HEIGHT:
            filtered_out_short = [
                obj["id"] for obj in objects
                if obj["id"] != target_id
                and _is_occluder_allowed(obj["id"], OCCLUDER_STOP_WORDS)
                and not _is_occluder_tall_enough(original_scene, obj, target, MIN_OCCLUDER_HEIGHT_RATIO)
            ]
            if filtered_out_short:
                print(f"[INFO] Filtered out {len(filtered_out_short)} occluders (too short, height ratio < {MIN_OCCLUDER_HEIGHT_RATIO}): {filtered_out_short}")

        filtered_out_narrow = [
            obj["id"] for obj in objects
            if obj["id"] != target_id
            and _is_occluder_allowed(obj["id"], OCCLUDER_STOP_WORDS)
            and not _is_occluder_wide_enough(original_scene, obj, target, MIN_OCCLUDER_FOOTPRINT_RATIO)
        ]
        if filtered_out_narrow:
            print(f"[INFO] Filtered out {len(filtered_out_narrow)} occluders (footprint ratio < {MIN_OCCLUDER_FOOTPRINT_RATIO}): {filtered_out_narrow}")
        
        if len(possible_occluders) == 0:
            print(f"[WARN] No occluders available for target {target_id} (after filtering)")
            continue
        
        # Select occluders according to configuration
        if PREFER_LARGER_OCCLUDER:
            # Prefer occluders larger than the target
            larger_occluders = [obj for obj in possible_occluders if _is_occluder_larger_than_target(original_scene, obj, target)]
            smaller_occluders = [obj for obj in possible_occluders if not _is_occluder_larger_than_target(original_scene, obj, target)]
            
            # Select larger occluders first; fill remaining slots with smaller ones if needed
            if len(larger_occluders) > 0:
                random.shuffle(larger_occluders)
                random.shuffle(smaller_occluders)
                selected_occluders = larger_occluders[:min(num_occluders_per_target, len(larger_occluders))]
                remaining = num_occluders_per_target - len(selected_occluders)
                if remaining > 0 and len(smaller_occluders) > 0:
                    selected_occluders.extend(smaller_occluders[:remaining])
                print(f"[INFO] Selected {len(selected_occluders)} occluders ({len([o for o in selected_occluders if _is_occluder_larger_than_target(original_scene, o, target)])} larger than target)")
            else:
                random.shuffle(possible_occluders)
                selected_occluders = possible_occluders[:min(num_occluders_per_target, len(possible_occluders))]
        else:
            # Randomly select occluders
            random.shuffle(possible_occluders)
            selected_occluders = possible_occluders[:min(num_occluders_per_target, len(possible_occluders))]
        
        print(f"\n[INFO] Processing target: {target_id} (distance: {target_distance:.2f}m)")
        print(f"[INFO] Selected {len(selected_occluders)} occluders: {[o['id'] for o in selected_occluders]}")
        
        for occ in selected_occluders:
            occ_id = occ["id"]
            success = False
            fail_stats = {
                "move_failed": 0,
                "occludes_failed": 0,
                "depth_failed": 0,
                "penetration_failed": 0,
                "contact_occ_failed": 0,
                "other_validation_failed": 0,
            }
            
            for attempt in range(NUM_TRIALS):
                # Create a fresh deep copy of the original scene each iteration to ensure full independence
                new_scene = copy.deepcopy(original_scene)
                occ_new = next(o for o in new_scene["objects"] if o["id"] == occ_id)
                tgt_new = next(o for o in new_scene["objects"] if o["id"] == target_id)

                # Move the occluder (target stays fixed); no occlusion degree distinction
                moved = move_occluder(new_scene, agent, tgt_new, occ_new, occlusion_ratio=None)
                if not moved:
                    fail_stats["move_failed"] += 1
                    continue

                # Basic occlusion check
                if not occludes(new_scene, agent, occ_new, tgt_new):
                    fail_stats["occludes_failed"] += 1
                    continue

                # Detailed constraint validation
                is_valid, validation_info = validate_occlusion(new_scene, agent, occ_new, tgt_new)
                if not is_valid:
                    if not validation_info["front_back_depth"]["valid"]:
                        fail_stats["depth_failed"] += 1
                    elif not validation_info["no_penetration"]["valid"]:
                        fail_stats["penetration_failed"] += 1
                    elif not validation_info["contact_plausibility_occ"]["valid"]:
                        fail_stats["contact_occ_failed"] += 1
                    else:
                        fail_stats["other_validation_failed"] += 1
                    continue
                
                # All checks passed; verify scene isolation (only occluder position was modified)
                isolated, diffs = verify_scene_isolation(original_scene, new_scene, occ_id)
                if not isolated:
                    print(f"  [WARN] Scene isolation check failed: {diffs}")
                    continue

                new_scene["occlusion"] = {
                    "occluder": occ_id,
                    "target": target_id,
                    "validation": validation_info
                }
                results_by_target[target_id].append(new_scene)
                success = True
                print(f"  [OK] Generated occlusion: {target_id} <- {occ_id}")
                break
            
            if not success:
                print(
                    f"  [WARN] Failed to generate occlusion: {target_id} <- {occ_id} after {NUM_TRIALS} attempts "
                    f"(move={fail_stats['move_failed']}, occludes={fail_stats['occludes_failed']}, "
                    f"depth={fail_stats['depth_failed']}, penetration={fail_stats['penetration_failed']}, "
                    f"contact={fail_stats['contact_occ_failed']}, other={fail_stats['other_validation_failed']})"
                )
    
    return results_by_target


def generate(json_path):
    """Generate occlusion scenes for all object pairs (original functionality; kept for backward compatibility)."""
    with open(json_path) as f:
        original_scene = json.load(f)  # save original scene to always start from a clean state

    camera_pos = _load_viewpoint(json_path)
    agent = camera_pos if camera_pos is not None else vec(original_scene["metadata"]["agent"]["position"])
    objects = original_scene["objects"]

    results = []

    for target in objects:
        for occ in objects:
            if occ["id"] == target["id"]:
                continue

            for _ in range(NUM_TRIALS):
                # Create a fresh deep copy of the original scene each iteration to ensure full independence.
                # Important: use original_scene (not scene) so every scene starts from the original state.
                new_scene = copy.deepcopy(original_scene)
                occ_new = next(o for o in new_scene["objects"] if o["id"] == occ["id"])
                tgt_new = next(o for o in new_scene["objects"] if o["id"] == target["id"])

                # Use new_scene to ensure only the current scene is modified
                moved = move_occluder(new_scene, agent, tgt_new, occ_new)
                if not moved:
                    continue

                # Basic occlusion check
                if not occludes(new_scene, agent, occ_new, tgt_new):
                    continue

                # Detailed constraint validation
                is_valid, validation_info = validate_occlusion(new_scene, agent, occ_new, tgt_new)
                if not is_valid:
                    continue

                # All checks passed; save the scene.
                # Note: each scene is a fully independent copy with only the occluder's position modified.
                new_scene["occlusion"] = {
                    "occluder": occ["id"],
                    "target": target["id"],
                    "validation": validation_info  # save validation info for debugging
                }
                results.append(new_scene)
                break

    return results


if __name__ == "__main__":
    import sys

    # Scene JSONs come from clean_scene_layout/
    # target_subgraph.json comes from rendered_scene/clean_scene_layout/
    layout_scene        = "/Users/zhangyue/Desktop/Holodeck/clean_scene_layout/"
    rendered_scene_root = "/Users/zhangyue/Desktop/Holodeck/rendered_scene/clean_scene_layout/"
    output_base         = "/Users/zhangyue/Desktop/Holodeck/occlusion_scenes_layout/"

    _excluded = {"target_subgraph.json", "object_attributes.json",
                 "camera_poses.json", "structure_proxy.json"}

    all_scenes_folder = os.listdir(layout_scene)
    all_scenes = []
    for scene_folder in all_scenes_folder:
        scene_path = os.path.join(layout_scene, scene_folder)
        if not os.path.isdir(scene_path):
            continue
        for file in os.listdir(scene_path):
            if file.endswith(".json") and file not in _excluded:
                all_scenes.append((scene_folder, os.path.join(scene_path, file)))

    print(f"Found {len(all_scenes)} scene files to process")
    
    for scene_folder, json_path in all_scenes:
        print(f"\n{'='*60}")
        print(f"Processing: {scene_folder} / {os.path.basename(json_path)}")
        print(f"{'='*60}")
        
        try:
            # If this scene already has output files under occlusion_scenes_layout, skip the entire scene_folder
            layout_out = os.path.join(output_base, scene_folder)
            if os.path.isdir(layout_out):
                try:
                    _names = os.listdir(layout_out)
                except OSError:
                    _names = []
                if any(
                    n.startswith("occlusion_") and n.endswith(".json") for n in _names
                ):
                    print(
                        f"[INFO] Skip scene: {layout_out} already contains occlusion_*.json "
                        f"(skip whole folder)"
                    )
                    continue

            scene_target_subgraph_path = os.path.join(rendered_scene_root, scene_folder, "target_subgraph.json")
            if not os.path.exists(scene_target_subgraph_path):
                print(f"[INFO] Skip scene: missing target_subgraph.json in {rendered_scene_root}/{scene_folder}")
                continue

            target_subgraph_path = scene_target_subgraph_path
            target_subgraph_data = load_target_subgraph_data(target_subgraph_path)
            target_ids = [
                target_id for target_id in target_subgraph_data.get("target_ids", [])
                if isinstance(target_id, str)
            ]
            target_subgraphs = target_subgraph_data.get("subgraphs", {})

            print(f"[INFO] Using scene-local target_subgraph: {scene_target_subgraph_path}")

            # Use the new target-grouped generation approach
            scenes_by_target = generate_by_target(
                json_path,
                num_occluders_per_target=10,
                target_ids=target_ids,
                target_subgraphs=target_subgraphs,
                rendered_scene_root=rendered_scene_root,
            )
            
            total_scenes = sum(len(scenes) for scenes in scenes_by_target.values())
            print(f"Generated {total_scenes} occlusion scenes across {len(scenes_by_target)} targets")
            
            if total_scenes == 0:
                print(f"[WARN] No occlusion scenes generated for {json_path}")
                continue
            
            # Create base output directory: occlusion_scenes/{scene_folder}/
            base_output_dir = os.path.join(output_base, scene_folder)
            os.makedirs(base_output_dir, exist_ok=True)
            
            # Save results grouped by target
            for target_id, scenes in scenes_by_target.items():
                if len(scenes) == 0:
                    continue
                
                # Sanitize target ID special characters for use in directory names
                target_safe = target_id.replace(" ", "_").replace("(", "").replace(")", "").replace("|", "_")
                
                # No longer create a separate directory per target.
                # Write JSONs directly to occlusion_scenes/{scene_folder}/.
                # Since filenames include target_safe, there are no name conflicts.
                target_dir = base_output_dir
                
                # Save all scenes for this target
                for s in scenes:
                    occlusion_info = s.get("occlusion", {})
                    occluder_id = occlusion_info.get("occluder", "unknown")
                    occlusion_ratio = occlusion_info.get("occlusion_ratio", None)
                    
                    # Sanitize occluder ID special characters for use in filenames
                    occluder_safe = occluder_id.replace(" ", "_").replace("(", "").replace(")", "").replace("|", "_")
                    
                    # Output filename: occlusion_{target}_{occluder}_{ratio}.json
                    if occlusion_ratio is not None:
                        ratio_str = f"{int(occlusion_ratio*100):02d}"
                        output_file = os.path.join(target_dir, f"occlusion_{target_safe}_{occluder_safe}_{ratio_str}.json")
                    else:
                        # Backward-compatible format (no occlusion ratio specified)
                        output_file = os.path.join(target_dir, f"occlusion_{target_safe}_{occluder_safe}.json")
                    
                    with open(output_file, "w") as f:
                        json.dump(s, f, indent=2)
                    print(f"  Saved: occlusion_{target_safe}_{occluder_safe}.json")
                
                print(f"[OK] Target {target_id}: {len(scenes)} scenes saved to {base_output_dir}")
            
            print(f"[OK] Processed {scene_folder}: {total_scenes} scenes saved to {base_output_dir}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {json_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"Processing complete! Output directory: {output_base}")
    print(f"{'='*60}")