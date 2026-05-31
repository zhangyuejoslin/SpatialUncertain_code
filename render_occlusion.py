import os
import time
from argparse import ArgumentParser
from pathlib import Path
import json
from typing import Any

import ai2thor
import compress_json
import numpy as np
import imageio.v2 as imageio

from ai2thor.controller import Controller
from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner

from ai2holodeck.constants import (
    THOR_COMMIT_ID,
    OBJATHOR_ASSETS_DIR,
)

import re
UID32 = re.compile(r"^[0-9a-f]{32}$")
# 3D export deps
import trimesh
from trimesh.transformations import euler_matrix
# Color extraction
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def _create_controller(args) -> Controller:
    """Bind WsgiServer on args.port; retry a few times if the port is still in TIME_WAIT."""
    for attempt in range(1, 6):
        try:
            return Controller(
                commit_id=THOR_COMMIT_ID,
                start_unity=False,
                port=args.port,
                scene="Procedural",
                gridSize=0.25,
                width=1024,
                height=1024,
                server_class=ai2thor.wsgi_server.WsgiServer,
                makeAgentsVisible=False,
                visibilityScheme="Distance",
                renderDepthImage=False,
                action_hook_runner=ProceduralAssetHookRunner(
                    asset_directory=args.asset_dir,
                    asset_symlink=True,
                    verbose=False,
                ),
            )
        except OSError as e:
            if "Address already in use" in str(e) and attempt < 5:
                print(
                    f"[WARN] Port {args.port} still in use (attempt {attempt}/5); "
                    f"waiting 2s for OS to release socket..."
                )
                time.sleep(2)
                continue
            raise


def save_png(path: str, rgb: np.ndarray):
    imageio.imwrite(path, rgb)


def save_mp4(path: str, frames, fps: int = 10):
    # imageio uses the ffmpeg backend; if ffmpeg is not installed, mp4 export may fail
    imageio.mimsave(path, frames, fps=fps)


def _to_xyz(v: dict | None) -> dict:
    v = v or {}
    return {
        "x": float(v.get("x", 0.0)),
        "y": float(v.get("y", 0.0)),
        "z": float(v.get("z", 0.0)),
    }


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _extract_camera_pose(event, camera_index: int = 0) -> dict:
    """
    Return camera pose from metadata if available.
    Fallback to agent pose with cameraHorizon.
    """
    meta = (event or {}).metadata if event is not None else {}
    tpc_list = meta.get("thirdPartyCameras", []) or []
    fov = _safe_float(meta.get("fov", 90.0), 90.0)

    if isinstance(tpc_list, list) and len(tpc_list) > camera_index:
        cam = tpc_list[camera_index] or {}
        return {
            "source": "third_party",
            "position": _to_xyz(cam.get("position")),
            "rotation": _to_xyz(cam.get("rotation")),
            "fov": _safe_float(cam.get("fieldOfView", fov), fov),
        }

    agent = meta.get("agent", {}) or {}
    rot = _to_xyz(agent.get("rotation"))
    rot["x"] = _safe_float(agent.get("cameraHorizon", rot.get("x", 0.0)), rot.get("x", 0.0))
    return {
        "source": "agent_fallback",
        "position": _to_xyz(agent.get("position")),
        "rotation": rot,
        "fov": fov,
    }


def _write_camera_poses_json(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _build_camera_pose_bundle(multiview_payload: dict, walkthrough_payload: dict) -> dict:
    return {
        "primary_sequence": "multiview",
        "camera_frame_for_modality_eval": "multiview",
        "multiview": multiview_payload,
        "walkthrough": walkthrough_payload,
    }



def _parse_occlusion_meta_from_scene_path(scene_path: str) -> dict[str, str] | None:
    scene_name = os.path.splitext(os.path.basename(scene_path))[0]
    if not scene_name.startswith("occlusion_"):
        return None

    body = scene_name[len("occlusion_"):]
    room_match = re.search(r"_[A-Za-z0-9]+$", body)
    if room_match is None:
        return None

    room_suffix = room_match.group(0)
    parts = body[: -len(room_suffix)].split(room_suffix + "_")
    if len(parts) != 2:
        return None

    target_token = parts[0] + room_suffix
    occluder_token = parts[1] + room_suffix
    room_name = room_suffix[1:].replace("_", " ")

    def _label(token: str) -> str:
        object_name = token[: -len(room_suffix)] if token.endswith(room_suffix) else token
        return f"{object_name} ({room_name})"

    return {
        "scene_type": "occlusion",
        "target_token": target_token,
        "occluder_token": occluder_token,
        "target_id": _label(target_token),
        "occluder_id": _label(occluder_token),
        "source_scene_json": os.path.abspath(scene_path),
    }


def _write_occlusion_meta_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _occlusion_meta_from_scene_json(scene_json: dict, scene_path: str) -> dict[str, Any] | None:
    """Build occlusion metadata from scene_json['occlusion'] when filename parsing fails."""
    occ = scene_json.get("occlusion") or {}
    if not isinstance(occ, dict):
        return None
    target_id = occ.get("target")
    occluder_id = occ.get("occluder")
    if not (isinstance(target_id, str) and isinstance(occluder_id, str)):
        return None
    return {
        "scene_type": "occlusion",
        "target_token": target_id,
        "occluder_token": occluder_id,
        "target_id": target_id,
        "occluder_id": occluder_id,
        "source_scene_json": os.path.abspath(scene_path),
    }


def _pick_oracle_frame(
    scene_json: dict,
    focus_token: str,
    multiview_payload: dict,
) -> int:
    """Return the multiview frame_idx whose camera forward most directly faces focus_token."""
    import math

    # Locate focus object position in scene JSON
    focus_pos = None
    for obj in scene_json.get("objects", []):
        obj_id = str(obj.get("id", ""))
        normalized = obj_id.replace(" (", "_").replace(")", "").replace(" ", "_")
        if obj_id == focus_token or normalized == focus_token:
            p = obj.get("position", {})
            focus_pos = (float(p.get("x", 0)), float(p.get("z", 0)))
            break

    if focus_pos is None:
        print(f"[WARN] _pick_oracle_frame: '{focus_token}' not found in scene objects, defaulting to frame 0")
        return 0

    frames = multiview_payload.get("frames", [])
    if not frames:
        return 0

    # All frames share the same camera position (agent rotates in place)
    cam_p = frames[0]["position"]
    cam_x, cam_z = float(cam_p.get("x", 0)), float(cam_p.get("z", 0))

    dx = focus_pos[0] - cam_x
    dz = focus_pos[1] - cam_z
    dist = math.sqrt(dx * dx + dz * dz)
    if dist < 1e-6:
        return 0
    dx /= dist
    dz /= dist

    best_idx, best_score = 0, -float("inf")
    for frame in frames:
        # AI2THOR convention: rotation.y = 0 → facing +Z
        yaw_deg = float(frame["rotation"].get("y", 0))
        yaw_rad = math.radians(yaw_deg)
        score = math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dz
        print(f"[DBG] oracle frame {frame['frame_idx']}: yaw={yaw_deg:.1f}° score={score:.3f}")
        if score > best_score:
            best_score = score
            best_idx = int(frame["frame_idx"])

    print(f"[OK] Oracle frame: view_{best_idx:03d}.png (facing '{focus_token}', dot={best_score:.3f})")
    return best_idx

def find_asset_mesh_path(asset_dir: str, asset_id: str) -> str | None:
    """
    Try to find the mesh file for a given assetId under OBJATHOR_ASSETS_DIR.
    Asset directory layouts vary across versions, so this uses common rules with
    a fallback recursive search.
    """
    root = Path(asset_dir)

    # Common candidates: file named directly after assetId
    candidates = [
        root / f"{asset_id}.glb",
        root / f"{asset_id}.obj",
        root / f"{asset_id}.ply",
        root / asset_id / f"{asset_id}.glb",
        root / asset_id / f"{asset_id}.obj",
        root / asset_id / f"{asset_id}.ply",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    # Fallback: recursive search (slow on first run but always works)
    # Only search glb/obj/ply to avoid excessive slowness
    exts = (".glb", ".obj", ".ply")
    for ext in exts:
        hits = list(root.rglob(f"{asset_id}{ext}"))
        if hits:
            return str(hits[0])

    return None


def pose_to_matrix(pos: dict, rot: dict, scale: dict | None = None) -> np.ndarray:
    """
    pos: {"x","y","z"}  (meters)
    rot: {"x","y","z"}  (degrees, Unity-style)
    scale: {"x","y","z"} optional
    """
    rx = np.deg2rad(rot.get("x", 0.0))
    ry = np.deg2rad(rot.get("y", 0.0))
    rz = np.deg2rad(rot.get("z", 0.0))

    # Unity uses various rotation orders (ZXY / XYZ); in THOR metadata yaw(y) is most critical.
    # Using XYZ here; switch to ZXY or yaw-only if orientation appears incorrect.
    T = euler_matrix(rx, ry, rz, axes="sxyz")
    T[0, 3] = pos.get("x", 0.0)
    T[1, 3] = pos.get("y", 0.0)
    T[2, 3] = pos.get("z", 0.0)

    if scale is not None:
        S = np.eye(4)
        S[0, 0] = scale.get("x", 1.0)
        S[1, 1] = scale.get("y", 1.0)
        S[2, 2] = scale.get("z", 1.0)
        T = T @ S

    return T


def fit_scale_from_aabb(mesh, obj):
    """
    Compute the scale factor from THOR's AABB and the mesh's actual extents.
    Prefers volume-ratio method (more accurate); falls back to geometric mean.
    """
    aabb = obj.get("axisAlignedBoundingBox", None)
    if not aabb or "size" not in aabb:
        return 1.0

    target = np.array(
        [aabb["size"]["x"], aabb["size"]["y"], aabb["size"]["z"]],
        dtype=np.float32
    )
    ext = np.array(mesh.extents, dtype=np.float32)

    if np.any(ext <= 1e-8) or np.any(target <= 1e-8):
        return 1.0

    # Method 1: volume ratio (most accurate, suitable for isotropic scaling)
    target_volume = np.prod(target)
    mesh_volume = np.prod(ext)
    if mesh_volume > 1e-12:
        s_volume = np.cbrt(target_volume / mesh_volume)  # cube root
    else:
        s_volume = None

    # Method 2: geometric mean (more robust for non-uniform scaling)
    ratios = target / ext
    # Filter outliers (e.g. one axis ratio is 10x larger than the others)
    valid_ratios = ratios[(ratios > 1e-6) & (ratios < 1e6)]
    if len(valid_ratios) == 0:
        return 1.0
    
    s_geometric = float(np.exp(np.mean(np.log(valid_ratios))))  # geometric mean

    # Prefer volume ratio; fall back to geometric mean if the two diverge too much
    if s_volume is not None:
        if abs(s_volume - s_geometric) / max(s_volume, s_geometric) < 0.5:  # less than 50% difference
            s = s_volume
        else:
            s = s_geometric
    else:
        s = s_geometric

    # Relaxed clamp range: allow smaller values (mesh may be in cm, needing 100x reduction)
    # and larger values (mesh may be in mm)
    s_raw = s
    s = max(min(s, 100.0), 0.001)  # allow range 0.001 to 100

    # Warn if the scale was clamped
    if s != s_raw:
        print(f"[WARN] Scale clamped: {s_raw:.6f} -> {s:.6f}")
    
    return s


def export_scene_glb(event, asset_dir: str, out_glb: str):
    objects = event.metadata.get("objects", [])
    scene = trimesh.Scene()
    missing = []
    scale_stats = []  # for tracking scale statistics

    for i, obj in enumerate(objects):
        asset_id = obj.get("assetId", None)
        if not asset_id:
            continue

        aid = asset_id.lower()
        if not UID32.match(aid):
            continue  # skip Doorway_* etc.

        mesh_path = find_asset_mesh_path(asset_dir, aid)
        if mesh_path is None:
            missing.append(asset_id)
            continue

        try:
            loaded = trimesh.load(mesh_path, force=None)

            # GLB files often load as a Scene; merge all geometries into one mesh
            if isinstance(loaded, trimesh.Scene):
                geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
                if not geoms:
                    raise ValueError("GLB Scene has 0 geometry")
                mesh = trimesh.util.concatenate(geoms)
            else:
                mesh = loaded

        except Exception as e:
            print(f"[WARN] Failed to load mesh for assetId={asset_id} path={mesh_path}: {e}")
            missing.append(asset_id)
            continue

        # Get position: prefer AABB center (more accurate), otherwise use position field
        aabb = obj.get("axisAlignedBoundingBox", {})
        aabb_center = aabb.get("center", {})
        if aabb_center and all(k in aabb_center for k in ["x", "y", "z"]):
            pos = {
                "x": float(aabb_center.get("x", 0.0)),
                "y": float(aabb_center.get("y", 0.0)),
                "z": float(aabb_center.get("z", 0.0))
            }
        else:
            pos = obj.get("position", {"x": 0, "y": 0, "z": 0})
        
        rot = obj.get("rotation", {"x": 0, "y": 0, "z": 0})
        
        # Compute scale so the mesh size matches THOR's AABB size
        aabb = obj.get("axisAlignedBoundingBox", {})
        aabb_size = aabb.get("size", {})
        target_size = np.array([
            aabb_size.get("x", 0),
            aabb_size.get("y", 0),
            aabb_size.get("z", 0)
        ])
        mesh_extents = np.array(mesh.extents, dtype=np.float32)
        
        s = fit_scale_from_aabb(mesh, obj)
        
        # Debug: show raw mesh size and target size
        if i < 5:  # only print details for the first 5 objects
            print(f"[DBG] {i:02d} {asset_id[:8]}... mesh_extents={mesh_extents.round(2)} "
                  f"target_size={target_size.round(2)} calculated_scale={s:.6f}")
        
        # Translate mesh to the origin (relative to its bounding box centroid)
        # so that all subsequent transforms are applied relative to the mesh origin
        mesh = mesh.copy()
        mesh_centroid = mesh.bounding_box.centroid.copy()
        mesh.apply_translation(-mesh_centroid)

        # Build the transform matrix: scale first, then rotate, then translate
        # Scale matrix (local coordinate system)
        S = np.eye(4, dtype=np.float32)
        S[0, 0] = S[1, 1] = S[2, 2] = s

        # Rotation matrix (local coordinate system)
        rx = np.deg2rad(rot.get("x", 0.0))
        ry = np.deg2rad(rot.get("y", 0.0))
        rz = np.deg2rad(rot.get("z", 0.0))
        R = euler_matrix(rx, ry, rz, axes="sxyz")

        # Translation matrix (world coordinate system — move object to THOR-specified position)
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = pos.get("x", 0.0)
        T[1, 3] = pos.get("y", 0.0)
        T[2, 3] = pos.get("z", 0.0)

        # Combined transform: T_final = T @ R @ S
        # Applied to point p: T_final @ p = T @ (R @ (S @ p))
        # Order: scale -> rotate -> translate
        T_final = T @ R @ S

        # Debug info
        aabb_size = aabb.get("size", {})
        ext_world = (np.array(mesh.extents, dtype=np.float32) * s)
        target_size = np.array([
            aabb_size.get("x", 0),
            aabb_size.get("y", 0),
            aabb_size.get("z", 0)
        ])
        scale_stats.append(s)
        
        if np.any(target_size > 0):
            size_diff = np.abs(ext_world - target_size) / (target_size + 1e-6)
            max_diff = np.max(size_diff)
            if max_diff > 0.3:  # difference exceeds 30%
                print(f"[WARN] {i:02d} {asset_id[:8]}... size mismatch: target={target_size.round(2)} vs actual={ext_world.round(2)} (diff={max_diff:.1%})")
        
        print(f"[DBG] {i:02d} {asset_id[:8]}... scale={s:.3f} pos=({pos['x']:.2f},{pos['y']:.2f},{pos['z']:.2f})")
        
        scene.add_geometry(
            mesh,
            node_name=f'{obj.get("objectId", asset_id)}__{i}',
            geom_name=f'{asset_id}__{i}',
            transform=T_final
        )
    
    # Print scale statistics
    if scale_stats:
        scale_stats = np.array(scale_stats)
        print(f"[STATS] Scale statistics: min={scale_stats.min():.3f} max={scale_stats.max():.3f} "
              f"mean={scale_stats.mean():.3f} std={scale_stats.std():.3f}")
    
    print("[DBG] scene bounds:", scene.bounds)

    print(f"[DBG] scene.geometry={len(scene.geometry)}  scene.graph.nodes={len(scene.graph.nodes)}")

    if len(scene.geometry) == 0:
        raise ValueError("Can't export empty scenes! (No uid meshes found/loaded)")

    scene.export(out_glb)
    print(f"[OK] Exported scene GLB -> {out_glb}")

    if missing:
        uniq = sorted(set(missing))
        print(f"[WARN] Missing meshes for {len(uniq)} uid assetIds (show up to 20): {uniq[:20]}")



def add_topdown_camera_and_capture(controller: Controller, out_png: str):
    """
    Capture a top-down view using THOR's GetMapViewCameraProperties (suitable for room layouts).
    """
    e = controller.step("GetMapViewCameraProperties")
    cam = e.metadata["actionReturn"]

    # Add a third-party camera with UI overlay disabled
    controller.step(
        action="AddThirdPartyCamera",
        position=cam["position"],
        rotation=cam["rotation"],
        fieldOfView=cam.get("fieldOfView", 90),
        renderImage=True,  # ensure image is rendered
    )
    e2 = controller.step("Pass")
    # third_party_camera_frames is a list; take the first element
    frame = e2.third_party_camera_frames[0]
    save_png(out_png, frame)
    print(f"[OK] Saved topdown -> {out_png}")


def capture_multiview(
    controller: Controller,
    out_dir: str,
    n_views: int = 8,
    radius: float = 1.5,
    poses_out_json: str | None = None,
) -> dict:
    """
    Multi-view capture: rotate the agent in place and take n_views images.
    Uses horizontal camera angle (horizon=0) rather than a top-down view.
    Uses a third-party camera to avoid UI overlay.
    Note: assumes the agent is already at the correct position and orientation (set by caller).
    """
    mkdir(out_dir)

    # Verify current agent horizon is horizontal
    agent_info = controller.last_event.metadata.get("agent", {})
    current_horizon = agent_info.get("cameraHorizon", 0)
    current_pos = agent_info.get("position", {})
    current_rot = agent_info.get("rotation", {})

    # Force horizontal view if current horizon is not level
    if abs(current_horizon) > 1.0:  # allow 1-degree tolerance
        print(f"[WARN] capture_multiview: Horizon is {current_horizon:.1f}°, forcing to 0°")
        # Method 1: use TeleportFull to reset horizon
        controller.step(
            action="TeleportFull",
            x=current_pos.get("x", 0),
            y=current_pos.get("y", 0.9),
            z=current_pos.get("z", 0),
            rotation=current_rot,
            horizon=0,  # force horizontal view
            standing=True
        )
        controller.step("Pass")
        event_check = controller.step("Pass")
        new_horizon = event_check.metadata.get("agent", {}).get("cameraHorizon", 999)

        # Method 2: if TeleportFull fails, use LookUp/LookDown to adjust
        if abs(new_horizon) > 1.0:
            print(f"[WARN] TeleportFull failed, trying LookUp/LookDown...")
            if new_horizon > 0:
                controller.step("LookUp", degrees=new_horizon)
            else:
                controller.step("LookDown", degrees=abs(new_horizon))
            controller.step("Pass")
            event_check = controller.step("Pass")
            new_horizon = event_check.metadata.get("agent", {}).get("cameraHorizon", 999)
        
        if abs(new_horizon) < 1.0:
            print(f"[OK] Horizon adjusted from {current_horizon:.1f}° to {new_horizon:.1f}°")
        else:
            print(f"[ERROR] Failed to set horizon to 0° (got {new_horizon:.1f}°)")

    # Use third-party camera to avoid UI overlay
    # Retrieve current agent camera parameters
    e = controller.step("Pass")
    agent_info = e.metadata.get("agent", {})
    agent_pos = agent_info.get("position", {})
    agent_rot = agent_info.get("rotation", {})
    agent_horizon = agent_info.get("cameraHorizon", 0)
    fov = e.metadata.get("fov", 90)

    # Configure third-party camera to match agent camera (without UI)
    cam_pos = {
        "x": agent_pos.get("x", 0),
        "y": agent_pos.get("y", 0.9),
        "z": agent_pos.get("z", 0)
    }
    cam_rot = {
        "x": float(agent_horizon),  # horizon as pitch
        "y": float(agent_rot.get("y", 0)),
        "z": float(agent_rot.get("z", 0))
    }

    # Add the third-party camera
    controller.step(
        action="AddThirdPartyCamera",
        position=cam_pos,
        rotation=cam_rot,
        fieldOfView=float(fov)
    )
    controller.step("Pass")

    pose_records = []
    image_height = None
    image_width = None

    for i in range(n_views):
        # Get current agent state
        e = controller.step("Pass")
        agent_info = e.metadata.get("agent", {})
        agent_pos = agent_info.get("position", {})
        agent_rot = agent_info.get("rotation", {})
        agent_horizon = agent_info.get("cameraHorizon", 0)

        # Sync third-party camera position and rotation with the agent
        cam_pos = {
            "x": agent_pos.get("x", 0),
            "y": agent_pos.get("y", 0.9),
            "z": agent_pos.get("z", 0)
        }
        cam_rot = {
            "x": float(agent_horizon),
            "y": float(agent_rot.get("y", 0)),
            "z": float(agent_rot.get("z", 0))
        }

        # Update the third-party camera
        controller.step(
            action="UpdateThirdPartyCamera",
            position=cam_pos,
            rotation=cam_rot
        )
        e2 = controller.step("Pass")

        # Use the third-party camera frame (no UI overlay)
        if hasattr(e2, 'third_party_camera_frames') and e2.third_party_camera_frames:
            frame = e2.third_party_camera_frames[0]
        elif hasattr(e2, 'thirdPartyCameraFrames') and e2.thirdPartyCameraFrames:
            frame = e2.thirdPartyCameraFrames[0]
        else:
            # Fall back to agent camera if third-party camera is unavailable
            print(f"[WARN] Third-party camera not available, using agent camera for view {i}")
            frame = e2.frame
        
        frame_name = f"view_{i:03d}.png"
        frame_path = os.path.join(out_dir, frame_name)
        save_png(frame_path, frame)

        if hasattr(frame, "shape") and len(frame.shape) >= 2:
            image_height = int(frame.shape[0])
            image_width = int(frame.shape[1])

        cam_pose = _extract_camera_pose(e2, camera_index=0)
        pose_records.append({
            "frame_idx": i,
            "image_file": frame_name,
            "position": cam_pose["position"],
            "rotation": cam_pose["rotation"],
            "fov": cam_pose["fov"],
            "camera_source": cam_pose["source"],
        })
        
        # Rotate agent to capture the next viewpoint
        if i < n_views - 1:  # no rotation needed after the last frame
            controller.step("RotateRight", degrees=360 / n_views)
            # Check and reset horizon after rotation (RotateRight may change horizon)
            controller.step("Pass")
            agent_after_rotate = controller.last_event.metadata.get("agent", {})
            horizon_after = agent_after_rotate.get("cameraHorizon", 0)
            if abs(horizon_after) > 1.0:
                if horizon_after > 0:
                    controller.step("LookUp", degrees=horizon_after)
                else:
                    controller.step("LookDown", degrees=abs(horizon_after))
                controller.step("Pass")

    print(f"[OK] Saved multiview frames -> {out_dir}")
    payload = {
        "type": "multiview",
        "n_views": n_views,
        "image_width": image_width,
        "image_height": image_height,
        "frames": pose_records,
    }
    if poses_out_json:
        _write_camera_poses_json(poses_out_json, payload)
        print(f"[OK] Saved multiview camera poses -> {poses_out_json}")
    return payload


def record_walkthrough(
    controller: Controller,
    out_mp4: str,
    steps: int = 80,
    fps: int = 10,
    poses_out_json: str | None = None,
) -> dict:
    """
    Record a simple trajectory video using MoveAhead + RotateRight/Left.
    Uses horizontal view angle (horizon=0) to simulate first-person walking.
    Uses a third-party camera to avoid UI overlay.
    """
    frames = []

    # Verify current agent horizon
    agent_info = controller.last_event.metadata.get("agent", {})
    current_horizon = agent_info.get("cameraHorizon", 0)
    current_pos = agent_info.get("position", {})
    current_rot = agent_info.get("rotation", {})

    # Force horizontal view if current horizon is not level
    if abs(current_horizon) > 1.0:
        print(f"[WARN] record_walkthrough: Horizon is {current_horizon:.1f}°, forcing to 0°")
        # Keep current position, only adjust horizon
        controller.step(
            action="TeleportFull",
            x=current_pos.get("x", 0),
            y=current_pos.get("y", 0.9),
            z=current_pos.get("z", 0),
            rotation=current_rot,
            horizon=0,  # force horizontal view
            standing=True
        )
        controller.step("Pass")
        event_check = controller.step("Pass")
        new_horizon = event_check.metadata.get("agent", {}).get("cameraHorizon", 999)

        # If TeleportFull fails, use LookUp/LookDown to adjust
        if abs(new_horizon) > 1.0:
            print(f"[WARN] TeleportFull failed, trying LookUp/LookDown...")
            if new_horizon > 0:
                controller.step("LookUp", degrees=new_horizon)
            else:
                controller.step("LookDown", degrees=abs(new_horizon))
            controller.step("Pass")
            event_check = controller.step("Pass")
            new_horizon = event_check.metadata.get("agent", {}).get("cameraHorizon", 999)
        
        if abs(new_horizon) < 1.0:
            print(f"[OK] Walkthrough horizon set to {new_horizon:.1f}°")
        else:
            print(f"[ERROR] Failed to set walkthrough horizon to 0° (got {new_horizon:.1f}°)")
    else:
        print(f"[OK] Walkthrough using current horizon={current_horizon:.1f}°")

    # Use third-party camera to avoid UI overlay
    # Retrieve current agent camera parameters
    e = controller.step("Pass")
    agent_info = e.metadata.get("agent", {})
    agent_pos = agent_info.get("position", {})
    agent_rot = agent_info.get("rotation", {})
    agent_horizon = agent_info.get("cameraHorizon", 0)
    fov = e.metadata.get("fov", 90)

    # Configure third-party camera to match agent camera (without UI)
    cam_pos = {
        "x": agent_pos.get("x", 0),
        "y": agent_pos.get("y", 0.9),
        "z": agent_pos.get("z", 0)
    }
    cam_rot = {
        "x": float(agent_horizon),  # horizon as pitch
        "y": float(agent_rot.get("y", 0)),
        "z": float(agent_rot.get("z", 0))
    }

    # Add the third-party camera
    controller.step(
        action="AddThirdPartyCamera",
        position=cam_pos,
        rotation=cam_rot,
        fieldOfView=float(fov)
    )
    controller.step("Pass")

    # Save initial position (used for 360-degree rotation without movement)
    initial_pos = {
        "x": agent_pos.get("x", 0),
        "y": agent_pos.get("y", 0.9),
        "z": agent_pos.get("z", 0)
    }
    
    pose_records = []
    image_height = None
    image_width = None

    for t in range(steps):
        # Compute rotation angle (evenly distributed over 360 degrees)
        rotation_angle = (t / steps) * 360.0

        # Update third-party camera: fixed position, rotation only
        cam_pos = initial_pos.copy()  # keep position fixed
        cam_rot = {
            "x": float(agent_horizon),  # keep horizon level
            "y": float(rotation_angle),  # only rotate yaw (left/right)
            "z": float(agent_rot.get("z", 0))
        }

        # Update the third-party camera
        controller.step(
            action="UpdateThirdPartyCamera",
            position=cam_pos,
            rotation=cam_rot
        )
        e2 = controller.step("Pass")

        # Use the third-party camera frame (no UI overlay)
        if hasattr(e2, 'third_party_camera_frames') and e2.third_party_camera_frames:
            frame = e2.third_party_camera_frames[0]
        elif hasattr(e2, 'thirdPartyCameraFrames') and e2.thirdPartyCameraFrames:
            frame = e2.thirdPartyCameraFrames[0]
        else:
            # Fall back to agent camera if third-party camera is unavailable
            print(f"[WARN] Third-party camera not available, using agent camera for frame {t}")
            frame = e2.frame
        
        frames.append(frame)
        if hasattr(frame, "shape") and len(frame.shape) >= 2:
            image_height = int(frame.shape[0])
            image_width = int(frame.shape[1])

        cam_pose = _extract_camera_pose(e2, camera_index=0)
        pose_records.append({
            "frame_idx": t,
            "position": cam_pose["position"],
            "rotation": cam_pose["rotation"],
            "fov": cam_pose["fov"],
            "camera_source": cam_pose["source"],
        })
        
        # No need to move the agent; only rotate the third-party camera
        # This ensures the position stays fixed while performing a 360-degree rotation

    save_mp4(out_mp4, frames, fps=fps)
    print(f"[OK] Saved video -> {out_mp4}")
    payload = {
        "type": "walkthrough",
        "video_file": os.path.basename(out_mp4),
        "fps": fps,
        "steps": steps,
        "image_width": image_width,
        "image_height": image_height,
        "frames": pose_records,
    }
    if poses_out_json:
        _write_camera_poses_json(poses_out_json, payload)
        print(f"[OK] Saved walkthrough camera poses -> {poses_out_json}")
    return payload


def export_structure_proxy(event, out_json: str):
    objs = event.metadata.get("objects", [])
    non_uid = []
    for o in objs:
        aid = (o.get("assetId") or "").lower()
        if not aid or UID32.match(aid):
            continue
        non_uid.append({
            "assetId": o.get("assetId"),
            "objectId": o.get("objectId"),
            "objectType": o.get("objectType"),
            "position": o.get("position"),
            "rotation": o.get("rotation"),
            "aabb": o.get("axisAlignedBoundingBox"),
            "obb": o.get("objectOrientedBoundingBox"),
            "name": o.get("name"),
        })
    with open(out_json, "w") as f:
        json.dump(non_uid, f, indent=2)
    print(f"[OK] Saved structure proxies -> {out_json} (count={len(non_uid)})")


def extract_color_from_albedo(asset_dir: str, asset_id: str) -> dict | None:
    """
    Extract the dominant color (RGB) from an asset's albedo texture.
    Returns {"r": 0-255, "g": 0-255, "b": 0-255, "hex": "#rrggbb"} or None.
    """
    if not HAS_PIL:
        return None
    root = Path(asset_dir)
    # Try to locate albedo.jpg
    candidates = [
        root / asset_id / "albedo.jpg",
        root / f"{asset_id}/albedo.png",
        root / f"{asset_id}/textures/albedo.jpg",
    ]
    for path in candidates:
        if path.exists():
            try:
                img = Image.open(path)
                img = img.convert("RGB")
                # Resize to 64x64 to speed up computation
                img = img.resize((64, 64), Image.Resampling.LANCZOS)
                pixels = np.array(img).reshape(-1, 3)
                # Compute average RGB
                avg_rgb = pixels.mean(axis=0).astype(int)
                return {
                    "r": int(avg_rgb[0]),
                    "g": int(avg_rgb[1]),
                    "b": int(avg_rgb[2]),
                    "hex": f"#{avg_rgb[0]:02x}{avg_rgb[1]:02x}{avg_rgb[2]:02x}",
                }
            except Exception:
                continue
    return None


def export_object_attributes(scene_json, event, out_json: str, asset_dir: str = None):
    """
    Merge object attributes from the raw scene JSON and Thor metadata, then save the combined info.
    - scene_json["objects"]: attributes from layout generation (object_name, id, roomId, vertices, etc.)
    - event.metadata["objects"]: attributes from Thor (AABB, OBB, objectType, name, etc.)
    - asset_dir: asset directory, used to extract color from albedo textures
    """
    json_objects = {obj.get("id"): obj for obj in scene_json.get("objects", [])}
    thor_objects = {obj.get("objectId"): obj for obj in event.metadata.get("objects", [])}
    
    # Build mapping: prefer matching by objectId == id; fall back to nearest-position match
    id_to_thor = {}
    for jid, jobj in json_objects.items():
        if jid in thor_objects:
            id_to_thor[jid] = jid
        else:
            # Position-based matching (simple: find nearest)
            jpos = jobj.get("position", {})
            jx, jy, jz = jpos.get("x", 0), jpos.get("y", 0), jpos.get("z", 0)
            best_thor_id = None
            best_dist = float("inf")
            for tid, tobj in thor_objects.items():
                if tid in id_to_thor.values():
                    continue
                tpos = tobj.get("position", {}) or {}
                tx, ty, tz = tpos.get("x", 0), tpos.get("y", 0), tpos.get("z", 0)
                d = (jx - tx) ** 2 + (jy - ty) ** 2 + (jz - tz) ** 2
                if d < best_dist:
                    best_dist = d
                    best_thor_id = tid
            if best_thor_id and best_dist < 0.5**2:
                id_to_thor[jid] = best_thor_id
    
    # Merge attributes
    attributes = []
    for jid, jobj in json_objects.items():
        thor_id = id_to_thor.get(jid)
        thor_obj = thor_objects.get(thor_id) if thor_id else None

        attr = {
            # Attributes from scene JSON
            "id": jid,
            "object_name": jobj.get("object_name"),
            "assetId": jobj.get("assetId"),
            "roomId": jobj.get("roomId"),
            "position": jobj.get("position"),
            "rotation": jobj.get("rotation"),
            "vertices": jobj.get("vertices"),
            "material": jobj.get("material"),
            "layer": jobj.get("layer"),
            "kinematic": jobj.get("kinematic"),
        }
        
        # Attributes from Thor metadata (if matched)
        if thor_obj:
            attr.update({
                "thor_objectId": thor_obj.get("objectId"),
                "thor_objectType": thor_obj.get("objectType"),
                "thor_name": thor_obj.get("name"),
                "aabb": thor_obj.get("axisAlignedBoundingBox"),
                "obb": thor_obj.get("objectOrientedBoundingBox"),
                "receptacle": thor_obj.get("receptacle"),
                "pickupable": thor_obj.get("pickupable"),
                "openable": thor_obj.get("openable"),
                "moveable": thor_obj.get("moveable"),
            })
            
            # Extract shape from AABB size
            aabb = thor_obj.get("axisAlignedBoundingBox", {})
            aabb_size = aabb.get("size", {})
            if aabb_size:
                attr["shape"] = {
                    "length": aabb_size.get("x", 0),  # length (typically the largest)
                    "width": aabb_size.get("z", 0),   # width
                    "height": aabb_size.get("y", 0),  # height
                    "volume": aabb_size.get("x", 0) * aabb_size.get("y", 0) * aabb_size.get("z", 0),
                }
                # Sort: length >= width >= height
                dims = sorted([attr["shape"]["length"], attr["shape"]["width"], attr["shape"]["height"]], reverse=True)
                attr["shape"]["dimensions"] = {"longest": dims[0], "middle": dims[1], "shortest": dims[2]}
        
        # Extract color from albedo texture
        asset_id = jobj.get("assetId")
        if asset_dir and asset_id:
            color = extract_color_from_albedo(asset_dir, asset_id)
            if color:
                attr["color"] = color
        
        # Improve material: if null in JSON, try to infer from Thor or other sources
        if not attr.get("material") and thor_obj:
            # Thor metadata may not have a direct material field, but it can be inferred from objectType
            obj_type = thor_obj.get("objectType", "")
            if obj_type:
                attr["material"] = {"inferred_from_type": obj_type}
        
        attributes.append(attr)
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(attributes, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved object attributes -> {out_json} (count={len(attributes)})")





def build_uid_asset_index(asset_dir: str):
    """
    Scan asset_dir for glb/obj/ply files and build a uid(32hex) -> mesh_path index.
    """
    root = Path(asset_dir)
    uid_to_paths = {}

    exts = (".glb", ".obj", ".ply")

    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue

        name = p.stem.lower()
        parent = p.parent.name.lower()

        # Case 1: filename is the uid
        if UID32.match(name):
            uid_to_paths.setdefault(name, []).append(str(p))
            continue

        # Case 2: parent directory name is the uid (most common: <uid>/model.glb)
        if UID32.match(parent):
            uid_to_paths.setdefault(parent, []).append(str(p))
            continue

        # Case 3: some path component is a uid (fallback)
        for part in map(str.lower, p.parts):
            if UID32.match(part):
                uid_to_paths.setdefault(part, []).append(str(p))
                break

    def pick_best(paths):
        # Prefer glb
        for ext in (".glb", ".obj", ".ply"):
            for p in paths:
                if p.lower().endswith(ext):
                    return p
        return paths[0]

    uid_to_best = {uid: pick_best(ps) for uid, ps in uid_to_paths.items()}
    print(f"[DBG] uid index built: {len(uid_to_best)} assets found in {asset_dir}")
    return uid_to_best


def _find_clean_camera_poses(scene_path: str, clean_scenes_root: str) -> str | None:
    """Auto-find camera_poses.json from paired clean scene given an occlusion scene path.

    Occlusion path structure:
        .../occlusion_scenes/{scene_id}/{target}/occlusion_{target}_{occluder}.json
    Clean scene root structure:
        clean_scenes_root/{scene_id}/{room}/camera_poses.json
    """
    p = Path(scene_path)
    # Robustly infer scene_id from any occlusion path layout:
    #   .../<scene_id>/<file>.json
    #   .../<scene_id>/<target>/<file>.json
    scene_id_re = re.compile(r".+-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d+$")
    scene_id = None
    for part in p.parts:
        if scene_id_re.match(part):
            scene_id = part
            break
    if not scene_id:
        # Fallback: assume direct parent is the scene_id
        scene_id = p.parent.name
    clean_base = Path(clean_scenes_root) / scene_id
    if not clean_base.exists():
        print(f"[WARN] Clean scene root not found: {clean_base}")
        return None

    # Layout (most common in your data):
    #   clean_scenes_root/<scene_id>/camera_poses.json
    direct_candidate = clean_base / "camera_poses.json"
    if direct_candidate.exists() and direct_candidate.is_file():
        return str(direct_candidate)

    for child in sorted(clean_base.iterdir()):
        # Expected alternative layout:
        #   clean_scenes_root/<scene_id>/<room>/camera_poses.json
        if not child.is_dir():
            continue
        candidate = child / "camera_poses.json"
        if candidate.exists() and candidate.is_file():
            return str(candidate)
    print(f"[WARN] No camera_poses.json found under: {clean_base}")
    return None


def _read_ref_position(ref_camera_poses: str) -> dict | None:
    """Read agent position from a reference camera_poses.json (clean scene)."""
    try:
        with open(ref_camera_poses) as f:
            data = json.load(f)
        frames = (data.get("multiview") or {}).get("frames") or []
        if frames:
            return frames[0].get("position")
    except Exception as e:
        print(f"[WARN] Could not read ref camera poses {ref_camera_poses}: {e}")
    return None


def process_single_scene(controller, scene_path, output_dir, args, ref_camera_poses: str | None = None):
    """
    Process a single scene: load scene, generate multiview images, walkthrough video, etc.
    """
    try:
        # Load scene
        scene_json = compress_json.load(scene_path)

        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(scene_path)}")
        print(f"{'='*60}")

        # Ensure clean scene state: clear the previous scene before loading the new one
        print("[INFO] Clearing previous scene state...")
        try:
            # Method 1: try using reset() to reset the scene
            controller.reset()
            print("[OK] Scene reset via reset()")
        except Exception as e:
            print(f"[INFO] Reset() not available, trying empty house method: {e}")
            # Method 2: create a minimal empty scene to clear the state
            try:
                empty_house = {
                    "doors": [],
                    "metadata": {},
                    "objects": [],
                    "proceduralParameters": {},
                    "rooms": [],
                    "walls": [],
                    "windows": []
                }
                controller.step(action="CreateHouse", house=empty_house)
                print("[OK] Empty house created to clear scene")
                # Wait for the empty scene to finish loading
                for _ in range(5):
                    controller.step("Pass")
            except Exception as e2:
                print(f"[WARN] Empty house method failed, using Pass: {e2}")
                # Method 3: if both methods fail, use multiple Pass steps
                for _ in range(5):
                    controller.step("Pass")

        # Extra wait to ensure cleanup is complete
        for _ in range(3):
            controller.step("Pass")

        # 1) build house (this fully replaces the scene rather than stacking on top)
        event = controller.step(action="CreateHouse", house=scene_json)
        print("[OK] CreateHouse done")

        # Wait for the scene to fully load (multiple Pass steps to ensure complete loading)
        for _ in range(5):
            controller.step("Pass")

        # Disable UI overlay if supported
        try:
            controller.step(action="SetUIEnabled", enabled=False)
            print("[OK] UI overlay disabled")
        except Exception as e:
            print(f"[INFO] UI disable not supported or already disabled: {e}")

        # Ensure agent uses horizontal view angle (horizon=0)
        # For occlusion scenes: use ref position from clean scene to ensure alignment
        # Determine start position ─────────────────────────────────────────
        # Occlusion scenes: use ref position from paired clean scene so that
        # the camera is always at the exact same spot as the clean render.
        # Clean scenes (or occlusion without a ref): fall back to GetReachablePositions.
        ref_pos = _read_ref_position(ref_camera_poses) if ref_camera_poses else None
        if ref_pos:
            print(f"[OK] Using ref camera position from clean scene: {ref_pos}")
            start_pos = ref_pos
        else:
            rp = controller.step("GetReachablePositions").metadata["actionReturn"]
            if rp:
                valid_rp = [p for p in rp if 0.5 <= p.get("y", 0) <= 1.5]
                if not valid_rp:
                    print("[WARN] No valid reachable positions (Y in [0.5, 1.5]), using all positions")
                    valid_rp = rp
                start_pos = valid_rp[len(valid_rp) // 2]
                start_pos["y"] = 0.9
            else:
                print("[WARN] No reachable positions found, using default position")
                start_pos = {"x": 0, "y": 0.9, "z": 0}

        # Snap x/z to nearest 0.25 grid — AI2THOR requires grid-aligned positions.
        # GetReachablePositions in Holodeck scenes returns continuous coords, so we snap.
        _G = 0.25
        start_pos = {
            "x": round(start_pos["x"] / _G) * _G,
            "y": start_pos["y"],
            "z": round(start_pos["z"] / _G) * _G,
        }
        print(f"[INFO] Snapped start position to grid: x={start_pos['x']} y={start_pos['y']:.3f} z={start_pos['z']}")

        # TeleportFull always runs regardless of how start_pos was obtained ──
        for attempt in range(5):
            teleport_event = controller.step(
                action="TeleportFull",
                x=start_pos["x"],
                y=start_pos["y"],
                z=start_pos["z"],
                rotation={"x": 0, "y": 0, "z": 0},
                horizon=0,
                standing=True,
                forceAction=True,
            )
            if not teleport_event.metadata.get("lastActionSuccess"):
                err = teleport_event.metadata.get("errorMessage", "unknown")
                print(f"[ERROR] TeleportFull FAILED (attempt {attempt+1}/5): {err}")
                print(f"[ERROR] Target position: x={start_pos['x']:.3f} y={start_pos['y']:.3f} z={start_pos['z']:.3f}")
                if attempt == 4:
                    raise RuntimeError(
                        f"TeleportFull to ref position failed after 5 attempts. "
                        f"The occluder may be blocking the clean scene camera position. "
                        f"Position: {start_pos}"
                    )
                continue

            controller.step("Pass")
            event_check = controller.step("Pass")

            agent_info = event_check.metadata.get("agent", {})
            actual_pos = agent_info.get("position", {})
            actual_horizon = agent_info.get("cameraHorizon", 999)
            actual_y = actual_pos.get("y", 0)

            if actual_y > 2.5:
                print(f"[ERROR] Agent Y too high ({actual_y:.2f}), retrying with y=0.9...")
                start_pos["y"] = 0.9
                continue

            if abs(actual_horizon) < 1.0:
                print(f"[OK] Agent at ({actual_pos.get('x',0):.2f}, {actual_y:.2f}, {actual_pos.get('z',0):.2f}) horizon={actual_horizon:.1f}°")
                break
            else:
                if attempt < 4:
                    print(f"[WARN] Horizon not set correctly (got {actual_horizon:.1f}°), retrying... ({attempt+1}/5)")
                    if actual_horizon > 0:
                        controller.step("LookUp", degrees=actual_horizon)
                    else:
                        controller.step("LookDown", degrees=abs(actual_horizon))
                    controller.step("Pass")
                else:
                    print(f"[ERROR] Failed to set horizon=0 after 5 attempts (got {actual_horizon:.1f}°)")

        # 2) multiview images
        multiview_pose_payload = capture_multiview(
            controller,
            os.path.join(output_dir, "multiview"),
            n_views=8,
        )

        # 3) walkthrough video
        walkthrough_pose_payload = record_walkthrough(
            controller,
            os.path.join(output_dir, "walkthrough.mp4"),
            steps=args.video_steps,
            fps=args.fps,
        )

        # 3.1) unified camera poses (single source of truth for modality-aligned eval)
        camera_pose_bundle = _build_camera_pose_bundle(
            multiview_payload=multiview_pose_payload,
            walkthrough_payload=walkthrough_pose_payload,
        )
        _write_camera_poses_json(
            os.path.join(output_dir, "camera_poses.json"),
            camera_pose_bundle,
        )
        print(f"[OK] Saved unified camera poses -> {os.path.join(output_dir, 'camera_poses.json')}")
        
        occlusion_meta = _parse_occlusion_meta_from_scene_path(scene_path)
        if occlusion_meta is None:
            occlusion_meta = _occlusion_meta_from_scene_json(scene_json, scene_path)
        if occlusion_meta is not None:
            # Pick the multiview frame that most directly faces the occluder
            focus_token = occlusion_meta.get("occluder_token") or occlusion_meta.get("target_token")
            oracle_idx = (
                _pick_oracle_frame(scene_json, focus_token, multiview_pose_payload)
                if focus_token else 0
            )
            occlusion_meta["oracle_frame"] = {
                "sequence": "multiview",
                "frame_idx": oracle_idx,
                "image_file": f"multiview/view_{oracle_idx:03d}.png",
            }
            _write_occlusion_meta_json(
                os.path.join(output_dir, "occlusion_meta.json"),
                occlusion_meta,
            )
            print(f"[OK] Saved occlusion metadata -> {os.path.join(output_dir, 'occlusion_meta.json')}")

        # 4) export structure proxy (optional)
        try:
            export_structure_proxy(event, os.path.join(output_dir, "structure_proxy.json"))
        except Exception as e:
            print(f"[WARN] Failed to export structure proxy: {e}")

        # 5) export object attributes (JSON + Thor metadata merged)
        try:
            export_object_attributes(
                scene_json, 
                event, 
                os.path.join(output_dir, "object_attributes.json"),
                asset_dir=args.asset_dir
            )
        except Exception as e:
            print(f"[WARN] Failed to export object attributes: {e}")

        print(f"[OK] Completed: {os.path.basename(scene_path)}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to process {scene_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_output_dir(scene_path, output_base, custom_out_dir=None):
    """
    Determine the output directory based on the scene path.
    Convention: for paths under occlusion_scenes, e.g.
      /.../occlusion_scenes/a_bedroom-2026-02-02-15-20-11-693081/bed-0_bedroom/occlusion_....json
    Output structure: exports_occlusion/{scene_folder}/{sub_folder}/{scene_name}/
    Mirrors the input directory hierarchy.
    """
    if custom_out_dir:
        return custom_out_dir
    
    scene_path_normalized = os.path.normpath(scene_path)
    p = Path(scene_path_normalized)
    parts = set(p.parts)

    # When input files come from occlusion_scenes* layouts, users typically want:
    #   rendered_scene/<scene_id>/<occlusion_target_occluder>/
    # instead of adding an extra intermediate directory like:
    #   rendered_scene/occlusion_scenes/<scene_id>/...
    if ("occlusion_scenes" in parts) or ("occlusion_scenes_layout" in parts):
        scene_id_re = re.compile(r".+-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d+$")
        scene_id = None
        for part in p.parts:
            if scene_id_re.match(part):
                scene_id = part
                break
        if scene_id is not None:
            scene_stem = p.stem  # e.g. occlusion_bench-0_bedroom_dresser-0_bedroom
            # If filename ends with a ratio suffix like "_05", strip it.
            scene_stem = re.sub(r"_\d{2}$", "", scene_stem)
            return os.path.join(output_base, scene_id, scene_stem)

    scene_dir = os.path.dirname(scene_path_normalized)  # 2-level: .../gen_scenes1/<scene_id>  or 3-level: .../<scene_id>/<sub_folder>
    scene_parent_dir = os.path.dirname(scene_dir)        # 2-level: .../gen_scenes1  or 3-level: .../<scene_id>

    group_folder = os.path.basename(scene_parent_dir)
    sub_folder = os.path.basename(scene_dir)
    scene_name = os.path.splitext(os.path.basename(scene_path))[0]

    # Detect timestamped scene ids like: a_bedroom-2026-02-02-15-24-49-813953
    scene_id_re = re.compile(r".+-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d+$")

    # Layout A (2-level): .../gen_scenes1/<scene_id>/<scene_name>.json
    # You probably want: exports_occlusion/gen_scenes1/<scene_id>/multiview (ignore scene_name).
    if scene_id_re.match(sub_folder):
        # If input files are under ".../occlusion_scenes_layout/<scene_id>/...",
        # we generally don't want to keep the intermediate "occlusion_scenes_layout" layer.
        if group_folder == "occlusion_scenes_layout":
            return os.path.join(output_base, sub_folder)
        return os.path.join(output_base, group_folder, sub_folder)

    # Layout B (3-level / redundant): .../<scene_id>/<sub_folder>/<sub_folder>.json
    # Avoid redundant output like .../<scene_id>/<sub_folder>/<sub_folder>/multiview.
    if scene_name == sub_folder:
        return os.path.join(output_base, group_folder)

    # Default: .../<scene_id>/<sub_folder>/<scene_name>.json  -> exports_occlusion/<scene_id>/<scene_name>/
    return os.path.join(output_base, group_folder, scene_name)


def is_scene_already_exported(output_dir):
    """
    Check whether a scene has already been exported.
    Criteria: output directory exists and contains key files (walkthrough.mp4 and multiview directory).
    """
    if not os.path.exists(output_dir):
        return False

    # Check whether key files exist
    walkthrough_path = os.path.join(output_dir, "walkthrough.mp4")
    multiview_dir = os.path.join(output_dir, "multiview")

    has_walkthrough = os.path.exists(walkthrough_path) and os.path.getsize(walkthrough_path) > 0
    has_multiview = os.path.exists(multiview_dir) and os.path.isdir(multiview_dir)

    # If the multiview directory exists, verify it contains image files
    if has_multiview:
        multiview_files = [f for f in os.listdir(multiview_dir) if f.endswith('.png')]
        has_multiview = len(multiview_files) > 0

    # Scene is considered exported when both key files are present
    return has_walkthrough and has_multiview


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--scene",
        help=(
            "Path to a single scene JSON file, or a directory containing multiple JSONs.\n"
            "If a directory is given, all *.json files (recursively) will be processed "
            "in the same Unity/Holodeck play session, so you only need to press Play once."
        ),
        required=True,
    )
    parser.add_argument("--asset_dir", default=OBJATHOR_ASSETS_DIR)
    parser.add_argument("--out_dir", default=None, 
                        help="Output directory (default: auto-generated from scene path)")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--video_steps", type=int, default=80)
    parser.add_argument("--port", type=int, default=8200, 
                        help="Port for AI2Thor controller")
    parser.add_argument("--continuous", action="store_true",
                        help="Continuous mode: process all scenes in one Unity session (only press Play once)")
    parser.add_argument(
        "--restart-controller-per-scene",
        action="store_true",
        help=(
            "Manual mode only: stop and create a new Controller for every scene (re-binds the port each time). "
            "Default is to reuse one Controller for the whole batch so port 8200 is not taken twice."
        ),
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Run without waiting for Enter at any prompt (fully automatic).",
    )
    parser.add_argument(
        "--ref-camera-poses",
        default=None,
        help=(
            "Path to camera_poses.json from the paired clean scene. "
            "When set, the agent teleports to that exact position before rendering "
            "so clean and occlusion frames are perfectly aligned."
        ),
    )
    args = parser.parse_args()

    scene_path = args.scene
    if not os.path.exists(scene_path):
        print(f"[ERROR] Path not found: {scene_path}")
        return
    
    rendered_root = "/Users/zhangyue/Desktop/Holodeck/rendered_scene"
    # Default output base (keeps input group folder, e.g. clean_scene_layout/<scene_id>/...)
    output_base = rendered_root
    # For occlusion scenes, write results under a dedicated subtree:
    #   rendered_scene/occlusion_scene_layout/<scene_id>/occlusion_<target>_<occluder>/
    occlusion_output_base = os.path.join(rendered_root, "occlusion_scene_layout")

    # Paired clean camera poses live under rendered clean outputs.
    clean_scenes_root = "/Users/zhangyue/Desktop/Holodeck/rendered_scene/clean_scene_layout"
    
    # Determine the list of scene files to process
    scene_path_obj = Path(scene_path)
    if scene_path_obj.is_dir():
        # Batch mode: recursively find all json files
        # NOTE: Some folders also contain auxiliary jsons (camera_poses/object_attributes/etc.).
        # Only keep actual scene jsons.
        skip_names = {
            "target_subgraph.json",
            "camera_poses.json",
            "object_attributes.json",
            "structure_proxy.json",
            "occlusion_meta.json",
        }
        json_files = sorted(
            p
            for p in scene_path_obj.rglob("*.json")
            if p.is_file()
            and p.name not in skip_names
        )
        if not json_files:
            print(f"[ERROR] No *.json files found in directory (including subdirectories): {scene_path}")
            return
        print(f"[INFO] Found {len(json_files)} JSON scenes in {scene_path} (recursive)")
    else:
        # Single file mode
        if not scene_path.endswith(".json"):
            print(f"[ERROR] Scene file must be a JSON file: {scene_path}")
            return
        json_files = [scene_path_obj]
    
    # Batch-process all scenes
    success_count = 0
    failed_count = 0
    total_scenes = len(json_files)
    
    print(f"\n{'='*60}")
    print(f"Total scenes to process: {total_scenes}")
    if args.continuous:
        print("Mode: CONTINUOUS (one Unity session for all scenes)")
    elif args.restart_controller_per_scene:
        print("Mode: MANUAL (new Controller per scene; may need wait between scenes for port release)")
    else:
        print(
            "Mode: MANUAL (one Controller for all scenes in this run — press Play once, "
            "Enter between scenes)"
        )
    print(f"{'='*60}")
    
    skipped_count = 0
    controller = None

    # Continuous mode: create the Controller only once
    if args.continuous:
        print(f"\n[INFO] Creating Controller on port {args.port}...")
        print("[INFO] Please make sure Unity/Holodeck is running and press Play ONCE!")
        if not args.no_prompt:
            input("Press Enter after Unity is in Play mode...")
        try:
            controller = _create_controller(args)
            print("[OK] Controller created successfully")
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"[ERROR] Port {args.port} is already in use!")
                print("[HINT] Please:")
                print("  1. Close any existing Unity/Holodeck instances")
                print("  2. Or use --port <different_port> to use a different port")
                print("  3. Or kill the process using the port")
                return
            else:
                raise
    
    for idx, scene_file in enumerate(json_files, start=1):
        scene_file_str = str(scene_file)

        # Route occlusion outputs to rendered_scene/occlusion_scene_layout/.
        # Clean scenes keep the default structure under rendered_scene/.
        base_for_this_scene = (
            occlusion_output_base
            if os.path.basename(scene_file_str).startswith("occlusion_")
            else output_base
        )
        output_dir = get_output_dir(scene_file_str, base_for_this_scene, args.out_dir)
        
        print(f"\n{'='*60}")
        print(f"[SCENE {idx}/{total_scenes}]")
        print(f"Scene file: {scene_file_str}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")
        
        # Check whether the scene has already been exported
        if is_scene_already_exported(output_dir):
            print(f"[SKIP] Scene already exported (walkthrough.mp4 and multiview/ exist)")
            skipped_count += 1
            continue
        
        # Create output directory
        mkdir(output_dir)

        # Manual mode: by default create only one Controller for the whole batch
        # (avoids port binding failures for the second scene)
        if not args.continuous:
            print(f"\n[INFO] Ready to process scene {idx}/{total_scenes}.")
            if args.restart_controller_per_scene or controller is None:
                if not args.no_prompt:
                    input(
                        "Press Enter when Unity/Holodeck is running AND you have pressed Play "
                        "(or Ctrl+C to stop)..."
                    )
            else:
                if not args.no_prompt:
                    input(
                        "Press Enter when ready for the next scene (same Unity Play session; "
                        "or Ctrl+C to stop)..."
                    )

            if args.restart_controller_per_scene or controller is None:
                print(f"\n[INFO] Creating Controller on port {args.port}...")
                print("[INFO] Connecting to Unity/Holodeck...")
                try:
                    controller = _create_controller(args)
                    print("[OK] Controller created successfully")
                except OSError as e:
                    if "Address already in use" in str(e):
                        print(f"[ERROR] Port {args.port} is still in use after retries!")
                        print("[HINT] Please:")
                        print("  1. Close any existing Unity/Holodeck instances")
                        print("  2. Or use --port <different_port>")
                        print(
                            "  3. Or omit --restart-controller-per-scene "
                            "(default: reuse one Controller for the whole folder)"
                        )
                        failed_count += 1
                        if idx < total_scenes:
                            print(f"\n[INFO] Skipping scene {idx}/{total_scenes} due to port error.")
                            if not args.no_prompt:
                                input("Press Enter to continue to next scene (or Ctrl+C to stop)...")
                        continue
                    raise
        
        # Process the current scene
        ref_poses = args.ref_camera_poses
        if ref_poses is None:
            # Only occlusion scenes require paired clean camera poses.
            # Clean scenes can be rendered without alignment-by-ref poses.
            occlusion_meta = _parse_occlusion_meta_from_scene_path(scene_file_str)
            if occlusion_meta is not None:
                ref_poses = _find_clean_camera_poses(scene_file_str, clean_scenes_root)
                if ref_poses:
                    print(f"[INFO] Auto-detected clean scene camera poses: {ref_poses}")
                else:
                    print(f"[ERROR] Cannot find clean scene camera_poses.json for occlusion: {scene_file_str}")
                    print(f"[ERROR] Make sure the clean scene camera poses are rendered under: {clean_scenes_root}")
                    return
        success = process_single_scene(controller, scene_file_str, output_dir, args,
                                       ref_camera_poses=ref_poses)
        
        if success:
            success_count += 1
        else:
            failed_count += 1
        
        # Manual mode: close Controller immediately only when restart-per-scene is on;
        # otherwise close it once at the end
        if not args.continuous and args.restart_controller_per_scene and controller is not None:
            print("\n[INFO] Closing Controller...")
            try:
                controller.stop()
                print("[OK] Controller stopped")
            except Exception as e:
                print(f"[WARN] Error stopping controller: {e}")
            time.sleep(2)
            controller = None

        if not args.continuous and idx < total_scenes:
            print(
                f"\n[INFO] Scene {idx}/{total_scenes} finished "
                f"({success_count} ok, {failed_count} failed)."
            )
    
    if controller is not None:
        print("\n[INFO] Closing Controller...")
        try:
            controller.stop()
            print("[OK] Controller stopped")
        except Exception as e:
            print(f"[WARN] Error stopping controller: {e}")
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Success: {success_count}/{total_scenes}")
    print(f"  Failed: {failed_count}/{total_scenes}")
    print(f"  Skipped (already exported): {skipped_count}/{total_scenes}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
