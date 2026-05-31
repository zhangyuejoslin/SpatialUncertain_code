"""
Minimal pipeline:
1) Load objects + camera poses.
2) Filter target objects using camera visibility/uniqueness.
3) Build world graph only from filtered targets.
"""

import json
import math
from typing import Optional
import os

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Default single-scene paths (used when run_pipeline is called directly).
# DATA_PATH = "/nas-ssd2/yuezhang/Holodeck/3d_essential/gen_scenes/a_bedroom-2026-02-02-15-20-11-693081/a_bedroom/object_attributes.json"
# CAMERA_PATH = "/nas-ssd2/yuezhang/Holodeck/3d_essential/gen_scenes/a_bedroom-2026-02-02-15-20-11-693081/a_bedroom/camera_poses.json"
# SUBGRAPH_OUTPUT_PATH = "/nas-ssd2/yuezhang/Holodeck/3d_essential/gen_scenes/a_bedroom-2026-02-02-15-20-11-693081/a_bedroom/target_subgraph.json"

# Root for batch processing: iterate all scenes under this folder.
GEN_SCENES1_ROOT = "/Users/zhangyue/Desktop/Holodeck/rendered_scene/clean_scene_layout"
#GEN_SCENES1_ROOT = "/nas-ssd2/yuezhang/Holodeck/3d_essential/occlusion_scenes"

MIN_VOLUME = 0.01
NEAR_RATIO = 0.20
FAR_RATIO = 0.55
DIRECTION_MARGIN = 0.15
SIZE_MARGIN = 0.10
HORIZONTAL_DISP_MIN = 0.20
REL_DISTANCE_MIN = 0.50
REL_DISTANCE_MAX = 3.50
VERTICAL_OFFSET_MAX = 1.20
VERTICAL_CATEGORY_KEYWORDS = ("painting", "wall_shelf")
FRONT_DZ_MIN = 0.60

MIN_VISIBLE_DEPTH = 0.35
VISIBILITY_MIN_SCORE = 0.10
MAX_OCCLUSION_PENALTY = 0.60
MIN_UNIQUENESS_SCORE = 0.40
REQUIRE_SCENE_LEVEL_UNIQUENESS = True
MAX_OBJECT_FOOTPRINT = 1.5  # max object footprint on the XZ plane (meters); objects exceeding this are excluded
TOP_K_TARGETS_PER_CAMERA = 3
FRONT_BEHIND_NEAR_MAX_DIST = 2.2


def _vec(p: dict) -> tuple[float, float, float]:
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


def _volume(size: dict) -> float:
    return float(size.get("x", 0.0)) * float(size.get("y", 0.0)) * float(size.get("z", 0.0))


def _dist_xz(p1: dict, p2: dict) -> float:
    x1, _, z1 = _vec(p1)
    x2, _, z2 = _vec(p2)
    return math.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2)


def _aabb_overlap(a: dict, b: dict) -> bool:
    ca = a.get("center", {})
    sa = a.get("size", {})
    cb = b.get("center", {})
    sb = b.get("size", {})
    ax_min = ca.get("x", 0) - sa.get("x", 0) / 2
    ax_max = ca.get("x", 0) + sa.get("x", 0) / 2
    ay_min = ca.get("y", 0) - sa.get("y", 0) / 2
    ay_max = ca.get("y", 0) + sa.get("y", 0) / 2
    az_min = ca.get("z", 0) - sa.get("z", 0) / 2
    az_max = ca.get("z", 0) + sa.get("z", 0) / 2
    bx_min = cb.get("x", 0) - sb.get("x", 0) / 2
    bx_max = cb.get("x", 0) + sb.get("x", 0) / 2
    by_min = cb.get("y", 0) - sb.get("y", 0) / 2
    by_max = cb.get("y", 0) + sb.get("y", 0) / 2
    bz_min = cb.get("z", 0) - sb.get("z", 0) / 2
    bz_max = cb.get("z", 0) + sb.get("z", 0) / 2
    return (
        ax_min < bx_max and ax_max > bx_min
        and ay_min < by_max and ay_max > by_min
        and az_min < bz_max and az_max > bz_min
    )


def _infer_category(obj: dict) -> str:
    cat = obj.get("thor_objectType") or obj.get("object_name")
    if isinstance(cat, str):
        c = cat.strip().lower()
        if c and c not in {"undefined", "unknown", "none", "null"}:
            return c
    oid = obj.get("id") or obj.get("thor_objectId") or ""
    if isinstance(oid, str) and "-" in oid:
        return oid.split("-")[0].strip().lower()
    return "unknown"


def _rel_dir_world(p_subj: dict, p_other: dict) -> Optional[str]:
    x1, _, z1 = _vec(p_subj)
    x2, _, z2 = _vec(p_other)
    dx = x1 - x2
    dz = z1 - z2
    adx = abs(dx)
    adz = abs(dz)
    if max(adx, adz) < DIRECTION_MARGIN:
        return None
    if adx > adz + DIRECTION_MARGIN:
        return "left_of" if dx < 0 else "right_of"
    if adz > adx + DIRECTION_MARGIN:
        return "behind" if dz < 0 else "front_of"
    return None


def _scene_diag_xz(nodes: list[dict]) -> float:
    if len(nodes) < 2:
        return 1.0
    xs = [n["center"].get("x", 0.0) for n in nodes]
    zs = [n["center"].get("z", 0.0) for n in nodes]
    return max(math.hypot(max(xs) - min(xs), max(zs) - min(zs)), 1.0)


def _camera_basis(rotation: dict) -> dict:
    rx = math.radians(rotation.get("x", 0.0))
    ry = math.radians(rotation.get("y", 0.0))
    forward = _normalize((math.sin(ry) * math.cos(rx), -math.sin(rx), math.cos(ry) * math.cos(rx)))
    right = _normalize(_cross((0.0, 1.0, 0.0), forward))
    if _norm(right) < 1e-8:
        right = (1.0, 0.0, 0.0)
    up = _normalize(_cross(forward, right))
    return {"forward": forward, "right": right, "up": up}


def _object_radius(size: dict) -> float:
    sx, sy, sz = float(size.get("x", 0)), float(size.get("y", 0)), float(size.get("z", 0))
    return 0.5 * math.sqrt(sx * sx + sy * sy + sz * sz)


def load_and_filter_data(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    out = []
    for obj in data:
        aabb = obj.get("aabb") or {}
        if _volume(aabb.get("size", {})) >= MIN_VOLUME:
            out.append(obj)
    return out


def load_camera_data(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def build_object_nodes(data: list[dict]) -> tuple[list[dict], dict]:
    nodes = []
    id_to_aabb = {}
    for obj in data:
        aabb = obj.get("aabb") or {}
        oid = obj.get("id") or obj.get("objectId") or obj.get("thor_objectId") or ""
        node = {
            "id": oid,
            "category": _infer_category(obj),
            "center": aabb.get("center", obj.get("position", {})),
            "size": aabb.get("size", {}),
            "room": obj.get("roomId") or "",
        }
        nodes.append(node)
        id_to_aabb[oid] = aabb
    return nodes, id_to_aabb


def build_camera_nodes(camera_data: dict) -> tuple[str, list[dict]]:
    seq = camera_data.get("camera_frame_for_modality_eval") or camera_data.get("primary_sequence") or "multiview"
    frames = (camera_data.get(seq) or {}).get("frames", [])
    nodes = []
    for frame in frames if isinstance(frames, list) else []:
        if not isinstance(frame, dict):
            continue
        idx = int(frame.get("frame_idx", len(nodes)))
        nodes.append({
            "id": f"{seq}_{idx:03d}",
            "frame_idx": idx,
            "position": frame.get("position", {}),
            "rotation": frame.get("rotation", {}),
            "fov": float(frame.get("fov", 90.0)),
        })
    return seq, nodes


def select_targets(object_nodes: list[dict], camera_nodes: list[dict]) -> dict:
    scene_cat_counts = {}
    for obj in object_nodes:
        cat = obj.get("category", "unknown")
        scene_cat_counts[cat] = scene_cat_counts.get(cat, 0) + 1

    by_camera = {}
    selected_ids = set()
    for cam in camera_nodes:
        cid = cam["id"]
        cam_pos = _vec(cam["position"])
        basis = _camera_basis(cam["rotation"])
        right = basis["right"]
        up = basis["up"]
        forward = basis["forward"]
        half_fov = math.radians(cam["fov"]) * 0.5

        feats = {}
        visible_ids = []
        for obj in object_nodes:
            oid = obj["id"]
            rel = _sub(_vec(obj["center"]), cam_pos)
            x_cam = _dot(rel, right)
            y_cam = _dot(rel, up)
            z_cam = _dot(rel, forward)
            in_fov = z_cam > 1e-6 and abs(math.atan2(x_cam, z_cam)) <= half_fov
            apparent = _object_radius(obj["size"]) / max(z_cam, 1e-3)
            visible = z_cam > MIN_VISIBLE_DEPTH and in_fov
            feats[oid] = {
                "x_cam": x_cam,
                "y_cam": y_cam,
                "z_cam": z_cam,
                "horizontal_angle": abs(math.atan2(x_cam, max(z_cam, 1e-3))) if z_cam > 0 else 999.0,
                "apparent": apparent,
                "visible": visible,
                "occlusion_penalty": 0.0,
            }
            if visible:
                visible_ids.append(oid)

        for i in range(len(visible_ids)):
            a = visible_ids[i]
            for j in range(i + 1, len(visible_ids)):
                b = visible_ids[j]
                fa, fb = feats[a], feats[b]
                ax = fa["x_cam"] / max(fa["z_cam"], 1e-3)
                ay = fa["y_cam"] / max(fa["z_cam"], 1e-3)
                bx = fb["x_cam"] / max(fb["z_cam"], 1e-3)
                by = fb["y_cam"] / max(fb["z_cam"], 1e-3)
                dir_dist = math.hypot(ax - bx, ay - by)
                if dir_dist > 0.18:
                    continue
                if fa["z_cam"] < fb["z_cam"]:
                    fb["occlusion_penalty"] += max(0.0, 0.35 - dir_dist)
                else:
                    fa["occlusion_penalty"] += max(0.0, 0.35 - dir_dist)

        visible_by_cat = {}
        for obj in object_nodes:
            oid = obj["id"]
            if feats[oid]["visible"]:
                visible_by_cat.setdefault(obj["category"], []).append(oid)

        scored = []
        for obj in object_nodes:
            oid = obj["id"]
            cat = obj["category"]
            f = feats[oid]
            if not f["visible"]:
                continue
            max_footprint = max(float(obj["size"].get("x", 0)), float(obj["size"].get("z", 0)))
            if max_footprint > MAX_OBJECT_FOOTPRINT:
                continue
            same_cat = visible_by_cat.get(cat, [])
            uniq_view = 1.0 if len(same_cat) <= 1 else 0.0
            angle_score = max(0.0, 1.0 - f["horizontal_angle"] / (math.pi / 2))
            scale_score = min(1.0, f["apparent"] / 0.25)
            vis_score = 0.7 * angle_score + 0.3 * scale_score
            scene_unique = scene_cat_counts.get(cat, 0) == 1
            if vis_score < VISIBILITY_MIN_SCORE:
                continue
            if f["occlusion_penalty"] > MAX_OCCLUSION_PENALTY:
                continue
            if uniq_view < MIN_UNIQUENESS_SCORE:
                continue
            if REQUIRE_SCENE_LEVEL_UNIQUENESS and not scene_unique:
                continue
            score = 0.45 * vis_score + 0.40 * uniq_view + 0.15 * max(0.0, 1.0 - min(1.0, f["occlusion_penalty"]))
            scored.append({
                "target_id": oid,
                "category": cat,
                "score": round(score, 4),
                "visibility_score": round(vis_score, 4),
                "scene_unique": scene_unique,
                "occlusion_penalty": round(f["occlusion_penalty"], 4),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:TOP_K_TARGETS_PER_CAMERA]
        by_camera[cid] = top
        for item in top:
            selected_ids.add(item["target_id"])

    return {
        "by_camera": by_camera,
        "selected_target_ids": sorted(selected_ids),
        "meta": {
            "top_k_targets_per_camera": TOP_K_TARGETS_PER_CAMERA,
            "require_scene_level_uniqueness": REQUIRE_SCENE_LEVEL_UNIQUENESS,
        },
    }


def construct_world_graph(target_nodes: list[dict], all_nodes: list[dict], id_to_aabb: dict) -> dict:
    edges = {
        "left_of": [],
        "front_of": [],
    }
    pair_metadata = []

    for ni in target_nodes:
        for nj in all_nodes:
            if ni["id"] == nj["id"]:
                continue
            ci, cj = ni["center"], nj["center"]
            dist = _dist_xz(ci, cj)
            dx = float(ci.get("x", 0.0)) - float(cj.get("x", 0.0))
            dy = float(ci.get("y", 0.0)) - float(cj.get("y", 0.0))
            dz = float(ci.get("z", 0.0)) - float(cj.get("z", 0.0))
            within_dist = REL_DISTANCE_MIN <= dist <= REL_DISTANCE_MAX
            within_height = abs(dy) <= VERTICAL_OFFSET_MAX
            ni_cat = str(ni.get("category", "")).lower()
            nj_cat = str(nj.get("category", "")).lower()
            has_vertical_cat = any(k in ni_cat for k in VERTICAL_CATEGORY_KEYWORDS) or any(
                k in nj_cat for k in VERTICAL_CATEGORY_KEYWORDS
            )
            eligible = within_dist and within_height and (not has_vertical_cat)

            # Keep only canonical relation "left_of"
            if eligible and dx < -HORIZONTAL_DISP_MIN:
                edges["left_of"].append((ni["id"], nj["id"]))

            # Keep only canonical relation "front_of"
            if eligible and dz > FRONT_DZ_MIN:
                edges["front_of"].append((ni["id"], nj["id"]))

            if eligible:
                pair_metadata.append({
                    "target_id": ni["id"],
                    "referent_id": nj["id"],
                    "distance_xz": round(dist, 4),
                    "dx_world": round(dx, 4),
                    "dy_world": round(dy, 4),
                    "dz_world": round(dz, 4),
                    "target_center": ci,
                    "referent_center": cj,
                    "target_size": ni["size"],
                    "referent_size": nj["size"],
                    "target_volume": round(_volume(ni["size"]), 4),
                    "referent_volume": round(_volume(nj["size"]), 4),
                    "overlap_aabb": _aabb_overlap(id_to_aabb.get(ni["id"], {}), id_to_aabb.get(nj["id"], {})),
                })

    return {
        "target_nodes": target_nodes,
        "referent_nodes": all_nodes,
        "edges": edges,
        "pair_metadata": pair_metadata,
        "meta": {
            "horizontal_disp_min": HORIZONTAL_DISP_MIN,
            "rel_distance_min": REL_DISTANCE_MIN,
            "rel_distance_max": REL_DISTANCE_MAX,
            "vertical_offset_max": VERTICAL_OFFSET_MAX,
            "vertical_category_keywords": list(VERTICAL_CATEGORY_KEYWORDS),
            "direction_margin": DIRECTION_MARGIN,
            "front_dz_min": FRONT_DZ_MIN,
        },
    }


def get_relations_for_target(world_graph: dict, target_id: str) -> dict:
    """
    Retrieve all world-graph relations involving a given target.

    Returns:
        {rel_type: [(target_id, other_id), ...], ...}
    """
    edges = world_graph.get("edges", {})
    pair_metadata = world_graph.get("pair_metadata", [])
    outgoing = {}
    for rel_type, pairs in edges.items():
        outgoing[rel_type] = [(a, b) for (a, b) in pairs if a == target_id]
    target_pairs = [x for x in pair_metadata if x.get("target_id") == target_id]
    return {
        "relations": outgoing,
        "pair_metadata": target_pairs,
    }


def build_target_subgraph(result: dict) -> dict:
    selected = result.get("targets", {}).get("selected_target_ids", [])
    world_graph = result.get("world_graph", {})
    subgraphs = {}
    for target_id in selected:
        subgraphs[target_id] = get_relations_for_target(world_graph, target_id)
    return {
        "target_ids": selected,
        "subgraphs": subgraphs,
        "meta": {
            "num_targets": len(selected),
            "camera_sequence": result.get("meta", {}).get("camera_sequence"),
        },
    }


def save_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_pipeline(data_path: Optional[str] = None, camera_path: Optional[str] = None) -> dict:
    data = load_and_filter_data(data_path or DATA_PATH)
    all_nodes, id_to_aabb = build_object_nodes(data)
    camera_data = load_camera_data(camera_path or CAMERA_PATH)
    camera_sequence, camera_nodes = build_camera_nodes(camera_data)

    # Step 1: filter targets (camera is only used here)
    targets = select_targets(all_nodes, camera_nodes)

    # Remove targets whose category appears more than once in the selected set
    # (e.g. painting-0 and painting-1 both selected → ambiguous when one is occluded)
    id_to_cat = {n["id"]: n["category"] for n in all_nodes}
    cat_counts: dict = {}
    for oid in targets["selected_target_ids"]:
        cat = id_to_cat.get(oid, "unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    deduped = [oid for oid in targets["selected_target_ids"] if cat_counts[id_to_cat.get(oid, "unknown")] == 1]
    targets["selected_target_ids"] = deduped

    selected = set(deduped)
    target_nodes = [n for n in all_nodes if n["id"] in selected]

    # Step 2: construct world graph with target nodes as subjects and all nodes as referents
    world_graph = construct_world_graph(target_nodes, all_nodes, id_to_aabb)
    return {
        "targets": targets,
        "world_graph": world_graph,
        "meta": {
            "camera_sequence": camera_sequence,
            "num_cameras_used_for_filter": len(camera_nodes),
        },
    }


def iter_scenes(root: str):
    """
    Yield (data_path, camera_path, output_path) for each scene under gen_scenes1.

    Convention:
      - object_attributes.json   -> data_path
      - camera_poses.json        -> camera_path
      - target_subgraph.json     -> output JSON (same folder)
    """
    for dirpath, _, filenames in os.walk(root):
        if "object_attributes.json" in filenames and "camera_poses.json" in filenames:
            folder = dirpath
            data_path = os.path.join(folder, "object_attributes.json")
            camera_path = os.path.join(folder, "camera_poses.json")
            out_path = os.path.join(folder, "target_subgraph.json")
            yield data_path, camera_path, out_path


def run_for_all_scenes(root: str = GEN_SCENES1_ROOT, dry_run: bool = False) -> None:
    """
    Batch mode:
      - Traverse gen_scenes1/*
      - For each scene folder with object_attributes.json + camera_poses.json,
        run the meta pipeline and write target_subgraph.json back into that folder.
    """
    count = 0
    for data_path, camera_path, out_path in iter_scenes(root):
        count += 1
        print(f"\n[{count}] Scene folder: {os.path.dirname(data_path)}")
        print(f"  data_path   = {data_path}")
        print(f"  camera_path = {camera_path}")
        print(f"  out_path    = {out_path}")
        if dry_run:
            continue
        result = run_pipeline(data_path=data_path, camera_path=camera_path)
        subgraph = build_target_subgraph(result)
        save_json(out_path, subgraph)
        print("  Filter meta:", result["targets"]["meta"])
        print("  Selected targets:", len(result["targets"]["selected_target_ids"]))
        print("  World graph target nodes:", len(result["world_graph"]["target_nodes"]))
        print("  World graph referent nodes:", len(result["world_graph"]["referent_nodes"]))
        print("  World thresholds:", result["world_graph"]["meta"])
        for rel_type, pairs in result["world_graph"]["edges"].items():
            print(f"    {rel_type}: {len(pairs)} pairs")

    print(f"\nDone. Processed {count} scene(s) under {root}.")


if __name__ == "__main__":
    # Batch over gen_scenes1; set dry_run=True first if you just want to list scenes.
    run_for_all_scenes(root=GEN_SCENES1_ROOT, dry_run=False)
