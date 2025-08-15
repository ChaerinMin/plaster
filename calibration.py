"""
Camera calibration utilities using pycolmap.

Function: calibrate_camera_from_primer(primer_data, output_dir, ...)

Accepted Primer data structures (auto-detected):
    1. Dict with key "members" (PRIMARY expected structure)
             primer_data = { "members": [ member_dict, ... ] }
             Each member_dict MUST contain at least:
                     name          -> sensor name (string)
                     frame_idx     -> (int) index within that sensor (may be None)
                     frame         -> numpy ndarray image OR string path (may be None)
                     timestamp     -> optional numeric timestamp (can be None)
                     sensor_data   -> optional additional payload (ignored here)
                     diff          -> optional field (ignored here)
             Each member is converted into a calibration frame if it has a usable
             image: member['frame'] (ndarray) OR member['frame'] (str path) or, as a
             fallback, member['sensor_data'] if it is an ndarray.
             Assigned frame id: f"{name}_{frame_idx}" if frame_idx not None else name.

    2. Dict with key "frames" -> iterable of frame objects.
    3. Iterable of frame objects directly.
             Frame object forms accepted:
                 - dict with keys: id|frame_id|timestamp (identifier) and either
                             * image: numpy ndarray (RGB/BGR) OR
                             * path : path to existing image file
                 - tuple/list: (frame_id, image_ndarray)

Behavior:
    - Writes any in-memory images to <output_dir>/images as JPEG.
    - Runs a minimal reconstruction with pycolmap to estimate intrinsics
        and extrinsics.
    - Returns a result dictionary with success flag, message, camera params,
        and per-image poses.

Edge cases handled:
    - pycolmap missing -> graceful failure.
    - Insufficient usable images (< min_images) -> early return.
    - Re-uses existing directory unless clear_previous=True.
    - Silently skips members/frames lacking image content.

Install dependency:
        pip install pycolmap
"""

from __future__ import annotations

import os
import shutil
import uuid
from typing import Any, Dict, List, Optional, Tuple


def _import_pycolmap():
    try:
        import pycolmap  # type: ignore
        return pycolmap
    except Exception:
        return None


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _normalize_frames(primer_data: Any) -> List[Dict[str, Any]]:
    """Convert heterogeneous Primer structures into a list of frame dicts.

    Output frame dict schema (may include extra fields from source):
        {
            'id': str,              # unique identifier
            'image': ndarray|None,  # in-memory image if available
            'path': str|None,       # on-disk image path if available
            'timestamp': any|None,
            'source_member': original_member_dict (for traceability)
        }
    Frames missing both 'image' and 'path' are discarded.
    """
    if primer_data is None:
        return []

    norm: List[Dict[str, Any]] = []

    # Case 1: New Primer structure with 'members'
    if isinstance(primer_data, dict) and "members" in primer_data and isinstance(primer_data["members"], (list, tuple)):
        members = primer_data["members"]
        try:
            import numpy as np  # type: ignore
        except Exception:  # pragma: no cover
            np = None  # type: ignore
        for m in members:
            if not isinstance(m, dict):
                continue
            name = m.get("name")
            frame_idx = m.get("frame_idx")
            ts = m.get("timestamp")
            frame_obj = m.get("frame")
            sensor_data = m.get("sensor_data")
            img = None
            path = None
            # Accept str path in frame
            if isinstance(frame_obj, str) and os.path.exists(frame_obj):
                path = frame_obj
            # ndarray?
            elif frame_obj is not None and 'numpy' in type(frame_obj).__module__:
                img = frame_obj
            # Fallback to sensor_data if ndarray
            elif sensor_data is not None and 'numpy' in type(sensor_data).__module__:
                img = sensor_data
            # Build id
            fid = str(name) if frame_idx is None else f"{name}_{frame_idx}"
            if img is None and (path is None or not os.path.exists(path)):
                continue  # skip unusable member
            norm.append({
                "id": fid,
                "image": img,
                "path": path,
                "timestamp": ts,
                "source_member": m,
            })
        return norm

    # Case 2: Dict with frames
    if isinstance(primer_data, dict) and "frames" in primer_data:
        frames = primer_data["frames"]
    else:
        frames = primer_data

    for f in frames:
        if isinstance(f, dict):
            fid = f.get("id") or f.get("frame_id") or f.get("timestamp") or uuid.uuid4()
            out = {"id": str(fid), "timestamp": f.get("timestamp"), "source_member": f}
            if "image" in f:
                out["image"] = f.get("image")
            if "path" in f:
                path = f.get("path")
                if isinstance(path, str) and os.path.exists(path):
                    out["path"] = path
            # If neither image nor valid path, skip
            if not (out.get("image") is not None or out.get("path") is not None):
                continue
            norm.append(out)
        elif isinstance(f, (list, tuple)) and len(f) >= 2:
            fid = f[0]
            norm.append({"id": str(fid), "image": f[1], "timestamp": None, "source_member": f})
    return norm


def _write_images(frames: List[Dict[str, Any]], image_dir: str) -> List[Tuple[str, str]]:
    """Persist in-memory images to disk. Returns list of (frame_id, file_path)."""
    import imghdr
    written: List[Tuple[str, str]] = []
    try:
        import cv2  # type: ignore
        has_cv2 = True
    except Exception:
        has_cv2 = False
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover
        return written

    for f in frames:
        frame_id = str(f.get("id") or f.get("frame_id") or f.get("timestamp") or uuid.uuid4())
        if "path" in f and isinstance(f["path"], str) and os.path.exists(f["path"]):
            written.append((frame_id, f["path"]))
            continue
        img = f.get("image")
        if img is None:
            continue
        if not isinstance(img, np.ndarray):
            continue
        out_path = os.path.join(image_dir, f"{frame_id}.jpg")
        try:
            if has_cv2 and img.ndim == 3 and img.shape[2] == 3:
                # Assume RGB, convert to BGR for cv2.imwrite
                import cv2  # type: ignore
                cv2.imwrite(out_path, img[:, :, ::-1])
            else:
                from PIL import Image  # type: ignore
                mode = "RGB" if img.ndim == 3 and img.shape[2] == 3 else "L"
                Image.fromarray(img.astype("uint8"), mode=mode).save(out_path, quality=95)
        except Exception:
            continue
        if os.path.exists(out_path) and imghdr.what(out_path) is not None:
            written.append((frame_id, out_path))
    return written


def calibrate_camera_from_primer(primer_data: Any,
                                 output_dir: str,
                                 camera_model: str = "PINHOLE",
                                 camera_params: Optional[List[float]] = None,
                                 clear_previous: bool = False,
                                 min_images: int = 5) -> Dict[str, Any]:
    """Run pycolmap camera calibration on Primer frame data.

    Returns a dict with keys: success, message, output_dir, camera_params,
    image_poses, num_registered_images (when successful).
    """
    pycolmap = _import_pycolmap()
    if pycolmap is None:
        return {"success": False, "message": "pycolmap not installed", "output_dir": output_dir}

    frames = _normalize_frames(primer_data)
    if len(frames) < min_images:
        return {"success": False, "message": f"Need >= {min_images} frames", "output_dir": output_dir}

    if clear_previous and os.path.isdir(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    _ensure_dir(output_dir)
    image_dir = _ensure_dir(os.path.join(output_dir, "images"))

    frame_path_list = _write_images(frames, image_dir)
    if len(frame_path_list) < min_images:
        return {"success": False, "message": f"Only {len(frame_path_list)} valid images", "output_dir": output_dir}

    # Ensure unique filenames (copy if duplicate names arise)
    used_names = set()
    rename_map = {}
    for idx, (fid, path) in enumerate(frame_path_list):
        base = os.path.basename(path)
        name = base
        if name in used_names:
            stem, ext = os.path.splitext(base)
            name = f"{stem}_{idx}{ext}"
            new_path = os.path.join(image_dir, name)
            if path != new_path:
                shutil.copy2(path, new_path)
            path = new_path
        used_names.add(name)
        rename_map[fid] = name

    try:
        reconstruction = None
        try:
            # Preferred high-level API (recent pycolmap)
            reconstruction = pycolmap.run_reconstruction(
                image_dir=image_dir,
                output_dir=output_dir,
                camera_model=camera_model,
                camera_params=camera_params or [],
            )
        except Exception:
            # Manual pipeline fallbacks
            database_path = os.path.join(output_dir, "database.db")
            extract_opts = pycolmap.ExtractionOptions()
            match_opts = pycolmap.MatchingOptions()
            pycolmap.extract_features(database_path, image_dir, extract_opts)
            pycolmap.match_exhaustive(database_path, match_opts)
            try:
                reconstruction = pycolmap.incremental_mapping(
                    database_path=database_path,
                    image_path=image_dir,
                    output_path=output_dir,
                )
            except Exception as e:
                return {"success": False, "message": f"Mapping failed: {e}", "output_dir": output_dir}

        if reconstruction is None or len(reconstruction.images) == 0:
            return {"success": False, "message": "No registered images", "output_dir": output_dir}

        # Camera intrinsics (first camera only)
        camera_params_out = {}
        if reconstruction.cameras:
            for cam_id, cam in reconstruction.cameras.items():
                camera_params_out = {
                    "camera_id": cam_id,
                    "model": cam.model,
                    "width": cam.width,
                    "height": cam.height,
                    "params": list(cam.params),
                }
                break

        inv_map = {v: k for k, v in rename_map.items()}
        image_poses = {}
        for image_id, image in reconstruction.images.items():
            orig_id = inv_map.get(image.name, image.name)
            image_poses[orig_id] = {
                "image_id": image_id,
                "qvec": list(image.qvec),
                "tvec": list(image.tvec),
                "camera_id": image.camera_id,
            }

        return {
            "success": True,
            "message": f"Calibration succeeded ({len(image_poses)} images)",
            "output_dir": output_dir,
            "camera_params": camera_params_out,
            "image_poses": image_poses,
            "num_registered_images": len(image_poses),
        }
    except Exception as e:  # Catch-all
        return {"success": False, "message": f"Exception: {e}", "output_dir": output_dir}


__all__ = ["calibrate_camera_from_primer"]
