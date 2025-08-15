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
import torch
import numpy as np
import pycolmap

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
    failure_reasons = []  # (frame_id, reason)

    for f in frames:
        frame_id = str(f.get("id") or f.get("frame_id") or f.get("timestamp") or uuid.uuid4())
        if "path" in f and isinstance(f["path"], str) and os.path.exists(f["path"]):
            written.append((frame_id, f["path"]))
            continue
        img = f.get("image")
        if img is None:
            failure_reasons.append((frame_id, "no 'image' key / value is None and no valid 'path' provided"))
            continue
        # Convert torch tensors to numpy if necessary
        if isinstance(img, torch.Tensor):
            # Move to CPU, detach
            t = img.detach().cpu()
            # If 3D tensor and first dim is channels (C,H,W)
            if t.ndim == 3 and t.shape[0] in (1, 3, 4):
                t = t.permute(1, 2, 0)  # HWC
            elif t.ndim == 2:
                pass  # HW
            elif t.ndim == 3 and t.shape[2] in (1, 3, 4):
                # Already HWC
                pass
            else:
                failure_reasons.append((frame_id, f"unsupported tensor shape {tuple(t.shape)}"))
                continue
            # Convert dtype
            if t.dtype.is_floating_point:
                t = t.clamp(0, 1) * 255.0
                arr = t.to(torch.uint8).numpy()
            else:
                # Clamp to 0-255 then cast
                t = torch.clamp(t, 0, 255)
                arr = t.to(torch.uint8).numpy()
            img = arr
        
        if not isinstance(img, np.ndarray):
            failure_reasons.append((frame_id, f"image object not numpy ndarray after conversion (type={type(img)})"))
            continue
        out_path = os.path.join(image_dir, f"{frame_id}.jpg")
        try:
            from PIL import Image  # type: ignore
            mode = "RGB" if img.ndim == 3 and img.shape[2] == 3 else "L"
            Image.fromarray(img.astype("uint8"), mode=mode).save(out_path, quality=95)
        except Exception as e:
            failure_reasons.append((frame_id, f"exception during write: {e}"))
            continue
        if os.path.exists(out_path):
            if imghdr.what(out_path) is not None:
                written.append((frame_id, out_path))
            else:
                failure_reasons.append((frame_id, "file written but failed basic image validation (imghdr returned None)"))
        else:
            failure_reasons.append((frame_id, "attempted write but file not found afterward"))
    if not written:
        print(f"ERROR: No images were written to '{image_dir}'. Provided frames: {len(frames)}")
        if failure_reasons:
            print("Failure details (frame_id: reason):")
            for fid, reason in failure_reasons:
                print(f" - {fid}: {reason}")
        print("Ensure frames contain numpy uint8 arrays or valid existing file 'path' entries.")
    return written

# Normalize reconstruction object across pycolmap versions
def _select_reconstruction(rec):
    if rec is None:
        return None
    if hasattr(rec, 'images') and hasattr(rec, 'cameras'):
        return rec
    if isinstance(rec, (list, tuple)):
        best = None
        best_count = -1
        for r in rec:
            if hasattr(r, 'images'):
                cnt = len(getattr(r, 'images', {}))
                if cnt > best_count:
                    best = r
                    best_count = cnt
        return best
    if isinstance(rec, dict):
        if 'images' in rec and 'cameras' in rec:
            return rec
        for _, v in rec.items():
            if hasattr(v, 'images') and hasattr(v, 'cameras'):
                return v
            if isinstance(v, dict) and 'images' in v and 'cameras' in v:
                return v
    return None

def calibrate_camera_from_primer(frame_data: Any,
                                 output_dir: str,
                                 camera_model: str = "OPENCV",
                                 camera_params: Optional[List[float]] = None,
                                 clear_previous: bool = False,
                                 min_images: int = 5,
                                 verbose: bool = False) -> Dict[str, Any]:
    """Run pycolmap camera calibration on frame data.

    Returns a dict with keys: success, message, output_dir, camera_params,
    image_poses, num_registered_images (when successful).
    """
    frames = _normalize_frames(frame_data)
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
        database_path = os.path.join(output_dir, "database.db")
        
        # Feature extraction
        pycolmap.extract_features(database_path=database_path, image_path=image_dir, camera_mode=pycolmap.CameraMode.AUTO, camera_model=camera_model)

        # Feature Matching
        pycolmap.match_exhaustive(database_path=database_path)
        
        # Reconstruction
        reconstruction = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=image_dir,
            output_path=output_dir,
            )
        
        # Bundle adjustment
        pycolmap.bundle_adjustment(reconstruction)

        norm_rec = _select_reconstruction(reconstruction)
        if norm_rec is None:
            debug_msg = "Reconstruction object format unsupported (no images)"
            if verbose:
                debug_details = {
                    "type": str(type(reconstruction)),
                    "keys": list(reconstruction.keys()) if isinstance(reconstruction, dict) else None,
                    "len": len(reconstruction) if isinstance(reconstruction, (list, tuple, dict)) else None,
                    "dir": [a for a in dir(reconstruction) if a in ("images", "cameras")] if reconstruction is not None else None,
                }
                debug_msg += f" | details={debug_details}"
            return {"success": False, "message": debug_msg, "output_dir": output_dir}

        # Access images / cameras generically
        images_dict = norm_rec.images if hasattr(norm_rec, 'images') else norm_rec.get('images', {})  # type: ignore
        if not images_dict:
            return {"success": False, "message": "No registered images", "output_dir": output_dir}
        cameras_dict = norm_rec.cameras if hasattr(norm_rec, 'cameras') else norm_rec.get('cameras', {})  # type: ignore

        camera_params_out = {}
        if isinstance(cameras_dict, dict):
            for cam_id, cam in cameras_dict.items():
                if hasattr(cam, 'model'):
                    camera_params_out = {
                        "camera_id": cam_id,
                        "model": cam.model,
                        "width": getattr(cam, 'width', None),
                        "height": getattr(cam, 'height', None),
                        "params": list(getattr(cam, 'params', [])),
                    }
                elif isinstance(cam, dict):
                    camera_params_out = {
                        "camera_id": cam_id,
                        "model": cam.get('model'),
                        "width": cam.get('width'),
                        "height": cam.get('height'),
                        "params": list(cam.get('params', [])),
                    }
                break

        inv_map = {v: k for k, v in rename_map.items()}
        image_poses = {}
        for image_id, image in images_dict.items():
            name = image.name if hasattr(image, 'name') else (image.get('name') if isinstance(image, dict) else str(image_id))
            orig_id = inv_map.get(name, name)
            if hasattr(image, 'qvec') and hasattr(image, 'tvec'):
                qvec = list(image.qvec)
                tvec = list(image.tvec)
                cam_id = getattr(image, 'camera_id', None)
            else:
                qvec = list(image.get('qvec', [])) if isinstance(image, dict) else []
                tvec = list(image.get('tvec', [])) if isinstance(image, dict) else []
                cam_id = image.get('camera_id') if isinstance(image, dict) else None
            image_poses[orig_id] = {"image_id": image_id, "qvec": qvec, "tvec": tvec, "camera_id": cam_id}

        return {"success": True,
                "message": f"Calibration succeeded ({len(image_poses)} images)",
                "output_dir": output_dir,
                "camera_params": camera_params_out,
                "image_poses": image_poses,
                "num_registered_images": len(image_poses)}
    except Exception as e:  # Catch-all
        return {"success": False, "message": f"Exception: {e}", "output_dir": output_dir}
