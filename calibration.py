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

def _prepare_image_array(img: Any) -> Optional[np.ndarray]:
    """Normalize an input (torch.Tensor | np.ndarray) into a uint8 array acceptable by PIL.

    Accepts shapes:
      - H x W (grayscale)
      - H x W x C (C in 1,3,4)
      - C x H x W (C in 1,3,4)  (will transpose)

    Handles dtypes float32/float64 (assumed in range 0-1 or 0-255) and integer types.
    Returns None if the array cannot be interpreted as an image.
    """
    if img is None:
        return None
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if not isinstance(img, np.ndarray):
        return None
    if img.ndim == 0:
        return None

    # Remove extraneous batch dimension if present: (1, C, H, W) or (1, H, W, C)
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]

    # CHW -> HWC if needed
    if img.ndim == 3:
        h, w, c = None, None, None
        # Determine if first dim is channel dimension
        if img.shape[0] in (1, 3, 4) and (img.shape[2] > 4 or img.shape[2] not in (1, 3, 4)):
            # Likely CHW because last dim does not look like channels
            img = np.transpose(img, (1, 2, 0))
        elif img.shape[0] in (1, 3, 4) and img.shape[2] in (1, 3, 4) and img.shape[0] <= img.shape[2]:
            # Ambiguous; assume CHW if height/width look large in positions 1/2
            if img.shape[1] > 8 and img.shape[2] > 8:
                img = np.transpose(img, (1, 2, 0))
        # After possible transpose, if channel dimension is first still, transpose
        if img.shape[0] in (1, 3, 4) and img.shape[2] > 4:
            img = np.transpose(img, (1, 2, 0))

    # If we have shape (C,H,W) still (rare fallback)
    if img.ndim == 3 and img.shape[0] in (1,3,4) and img.shape[-1] not in (1,3,4):
        img = np.transpose(img, (1,2,0))

    # Squeeze single-channel dimension if grayscale
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    # Convert dtype -> uint8
    if np.issubdtype(img.dtype, np.floating):
        # Heuristic: if max <= 1.0 assume 0-1 range
        max_val = float(np.nanmax(img)) if img.size else 0.0
        if max_val <= 1.0 + 1e-6:
            img = img * 255.0
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
    elif img.dtype != np.uint8:
        # Scale larger integer types
        info = np.iinfo(img.dtype) if np.issubdtype(img.dtype, np.integer) else None
        if info and info.max > 255:
            img = (img.astype(np.float32) / info.max) * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # Final sanity checks
    if img.ndim == 3 and img.shape[2] not in (3, 4):
        # Unexpected channel count
        return None
    if img.ndim not in (2, 3):
        return None
    return img


def _write_images(frames: List[Dict[str, Any]], image_dir: str) -> List[Tuple[int, str]]:
    # Write frames to files. frames are torch tensors or ndarrays in various layouts.
    written: List[Tuple[int, str]] = []
    for f in frames:
        frame_id = f.get("id")
        raw_img = f.get("image")
        img = _prepare_image_array(raw_img)
        if img is None:
            print(f"Skipping frame {frame_id}: unsupported image shape/type {getattr(raw_img, 'shape', None)}")
            continue
        out_path = os.path.join(image_dir, f"{frame_id}.jpg")
        try:
            from PIL import Image  # type: ignore
            Image.fromarray(img).save(out_path, quality=95)
        except Exception as e:
            print(f"Failed to write image {frame_id}: {e} (original shape {getattr(raw_img, 'shape', None)})")
            continue
        if os.path.exists(out_path):
            written.append((frame_id, out_path))
    return written

def calibrate_camera_from_primer(frame_data: Any,
                                 output_dir: str,
                                 clear_previous: bool = False,
                                 min_images: int = 5,
                                 verbose: bool = True) -> Dict[str, Any]:
    frames = [{"id": str(ctr).zfill(3), "image": f} for ctr, f in enumerate(frame_data)]
    if len(frames) < min_images:
        return {"success": False, "message": f"Need >= {min_images} frames", "output_dir": output_dir}

    if clear_previous and os.path.isdir(output_dir):
        print(f"Clearing previous output directory: {output_dir}")
        shutil.rmtree(output_dir, ignore_errors=True)

    image_dir = os.path.join(output_dir, "images")
    undistorted_image_dir = os.path.join(output_dir, "undistorted_images")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(undistorted_image_dir, exist_ok=True)

    frame_path_list = _write_images(frames, image_dir)
    if len(frame_path_list) < min_images:
        return {"success": False, "message": f"Only {len(frame_path_list)} valid images", "output_dir": output_dir}

    try:
        database_path = os.path.join(output_dir, "database.db")

        # We will follow a 2-step strategy
        # Step 1 assumes a single camera model for all cameras, extracts distortion parameters and undistorts the images
        # Step 2 takes the undistorted images and refines the camera parameters using a PINHOLE model

        # Feature extraction
        sift_options = pycolmap.SiftExtractionOptions()
        sift_options.max_num_features = 24000 # Maximize number of features

        # SIMPLE_RADIAL_FISHEYE, RADIAL_FISHEYE, OPENCV_FISHEYE, FOV, THIN_PRISM_FISHEYE, RAD_TAN_THIN_PRISM_FISHEYE: Use these camera models for fisheye lenses and note that all other models are not really capable of modeling the distortion effects of fisheye lenses. The FOV model is used by Google Project Tango (make sure to not initialize omega to zero).
        pycolmap.extract_features(database_path=database_path, image_path=image_dir, camera_mode=pycolmap.CameraMode.SINGLE, camera_model=pycolmap.RADIAL_FISHEYE)

        # Feature Matching
        pycolmap.match_exhaustive(database_path=database_path)
        
        # Reconstruction
        incremental_options = pycolmap.IncrementalPipelineOptions()
        # incremental_options.multiple_models = False # Avoid multiple models
        # incremental_options.max_num_models = 1
        incremental_options.ba_global_function_tolerance = 0.000001
        reconstruction = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=image_dir,
            output_path=output_dir,
            options=incremental_options
        )
        
        # SIMPLE_PINHOLE, PINHOLE: Use these camera models, if your images are undistorted a priori. These use one and two focal length parameters, respectively. Note that even in the case of undistorted images, COLMAP could try to improve the intrinsics with a more complex camera model.
        # SIMPLE_RADIAL, RADIAL: This should be the camera model of choice, if the intrinsics are unknown and every image has a different camera calibration, e.g., in the case of Internet photos. Both models are simplified versions of the OPENCV model only modeling radial distortion effects with one and two parameters, respectively.

        # OPENCV, FULL_OPENCV: Use these camera models, if you know the calibration parameters a priori. You can also try to let COLMAP estimate the parameters, if you share the intrinsics for multiple images. Note that the automatic estimation of parameters will most likely fail, if every image has a separate set of intrinsic parameters.

        
        # Undistort images
        # pycolmap.undistort_images(database_path=database_path, image_path=image_dir, output_path=output_dir)

        return "WIP"
        # return {"success": True,
        #         "message": f"Calibration succeeded ({len(image_poses)} images)",
        #         "output_dir": output_dir,
        #         "camera_params": camera_params_out,
        #         "image_poses": image_poses,
        #         "num_registered_images": len(image_poses)}
    except Exception as e:
        return {"success": False, "message": f"Exception: {e}", "output_dir": output_dir}
