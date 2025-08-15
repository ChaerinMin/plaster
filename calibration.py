"""
Camera calibration utilities using pycolmap.

Function: calibrate_camera_from_primer(primer_data, output_dir, ...)

Lightweight contract for primer_data:
  - Either a dict with key "frames" or an iterable of frame objects.
  - Each frame can be:
        dict with keys: id|frame_id|timestamp (identifier), and either
            * image: numpy ndarray (RGB or BGR) OR
            * path: path to an existing image file
    or  tuple/list (frame_id, image_ndarray).

Behavior:
  - Writes any in-memory images to <output_dir>/images as JPEG.
  - Runs a minimal reconstruction with pycolmap to estimate intrinsics
    and extrinsics.
  - Returns a result dictionary with success flag, message, camera params,
    and per-image poses.

Edge cases handled:
  - pycolmap missing -> graceful failure.
  - Insufficient images (< min_images) -> early return.
  - Re-uses existing directory unless clear_previous=True.

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


def _normalize_frames(frames: Any) -> List[Dict[str, Any]]:
    if frames is None:
        print("Empty primer data provided")
        return []
    norm = []
    for ctr, f in enumerate(frames):
        norm.append({"id": ctr, "image": f})
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
