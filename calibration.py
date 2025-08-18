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
import glob

MAX_SIFT_FEATURES=25000

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
    if not os.path.exists(image_dir):
        os.makedirs(image_dir, exist_ok=True)
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

    stage1_dir = os.path.join(output_dir, "stage1")
    stage2_dir = os.path.join(output_dir, "stage2")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)

    frame_path_list = _write_images(frames, os.path.join(stage1_dir, "images"))
    if len(frame_path_list) < min_images:
        return {"success": False, "message": f"Only {len(frame_path_list)} valid images", "output_dir": output_dir}

    # We will follow a 2-stage strategy
    # Stage 1 assumes a single camera model for all cameras, extracts distortion parameters and undistorts the images
    # Stage 2 takes the undistorted images and refines the camera parameters using a PINHOLE model
    try:
        print(f"Stage 1 (RADIAL_FISHEYE) calibration started")
        stage1_database_path = os.path.join(stage1_dir, "stage1.db")
        stage1_image_dir = os.path.join(stage1_dir, "images")

        sift_options = pycolmap.SiftExtractionOptions()
        sift_options.max_num_features = MAX_SIFT_FEATURES # Maximize number of features
        # SIMPLE_RADIAL_FISHEYE, RADIAL_FISHEYE, OPENCV_FISHEYE, FOV, THIN_PRISM_FISHEYE, RAD_TAN_THIN_PRISM_FISHEYE: Use these camera models for fisheye lenses and note that all other models are not really capable of modeling the distortion effects of fisheye lenses. The FOV model is used by Google Project Tango (make sure to not initialize omega to zero).
        pycolmap.extract_features(database_path=stage1_database_path, image_path=stage1_image_dir, camera_mode=pycolmap.CameraMode.SINGLE, camera_model='RADIAL_FISHEYE')

        pycolmap.match_exhaustive(database_path=stage1_database_path)
        
        # Reconstruction
        incremental_options = pycolmap.IncrementalPipelineOptions()
        incremental_options.multiple_models = False # Avoid multiple models
        incremental_options.max_num_models = 1
        incremental_options.ba_global_function_tolerance = 0.000001
        stage1_reconstruction = pycolmap.incremental_mapping(
            database_path=stage1_database_path,
            image_path=stage1_image_dir,
            output_path=stage1_dir,
            options=incremental_options
        )
        stage1_reconstruction[0].write(stage1_dir) # Write explicitly

        # Undistort images
        stage2_image_dir = os.path.join(stage2_dir, "images")
        undistort_options = pycolmap.UndistortCameraOptions()
        undistort_options.max_image_size = 1920 # PARAM
        pycolmap.undistort_images(output_path=stage2_image_dir, input_path=stage1_dir, image_path=stage1_image_dir)
        # Re-structure undistorted_image_dir
        file_names = os.listdir(os.path.join(stage2_image_dir, "images"))
        for file_name in file_names:
            print(f"Moving {file_name} to {stage2_image_dir}/")
            shutil.move(os.path.join(stage2_image_dir, "images", file_name), os.path.join(stage2_image_dir, file_name))
        shutil.rmtree(os.path.join(stage2_image_dir, "images"), ignore_errors=True)
        shutil.rmtree(os.path.join(stage2_image_dir, "sparse"), ignore_errors=True)
        shutil.rmtree(os.path.join(stage2_image_dir, "stereo"), ignore_errors=True)
        sh_files = glob.glob(os.path.join(stage2_image_dir, "run-*.sh"))
        for sh_file in sh_files:
            shutil.rmtree(sh_file, ignore_errors=True)
            
        # Print distortion parameters
        print(f"Radial distortion parameters: {stage1_reconstruction[0].camera.intrinsics}")

        print(f"Stage 1 (RADIAL_FISHEYE) calibration completed")
    except Exception as e:
        print(f"Stage 1 (fisheye) calibration failed: {e}. Not proceeding to Stage 2. Exiting.")
        return {"success": False, "message": f"Exception: {e}", "output_dir": output_dir} 
        
    try:
        print(f"Stage 2 (SIMPLE_PINHOLE) calibration started")
        stage2_database_path = os.path.join(stage2_dir, "stage2.db")

        sift_options = pycolmap.SiftExtractionOptions()
        sift_options.max_num_features = MAX_SIFT_FEATURES # Maximize number of features
        # SIMPLE_PINHOLE, PINHOLE: Use these camera models, if your images are undistorted a priori. These use one and two focal length parameters, respectively. Note that even in the case of undistorted images, COLMAP could try to improve the intrinsics with a more complex camera model.
        # OPENCV, FULL_OPENCV: Use these camera models, if you know the calibration parameters a priori. You can also try to let COLMAP estimate the parameters, if you share the intrinsics for multiple images. Note that the automatic estimation of parameters will most likely fail, if every image has a separate set of intrinsic parameters.        
        pycolmap.extract_features(database_path=stage2_database_path, image_path=stage2_image_dir, camera_mode=pycolmap.CameraMode.PER_IMAGE, camera_model='SIMPLE_PINHOLE')

        pycolmap.match_exhaustive(database_path=stage2_database_path)

        # Reconstruction
        incremental_options = pycolmap.IncrementalPipelineOptions()
        incremental_options.multiple_models = False # Avoid multiple models
        incremental_options.max_num_models = 1
        incremental_options.ba_global_function_tolerance = 0.000001
        stage2_reconstruction = pycolmap.incremental_mapping(
            database_path=stage2_database_path,
            image_path=stage2_image_dir,
            output_path=stage2_dir,
            options=incremental_options
        )
        stage2_reconstruction[0].write(stage2_dir)
        print(f"Stage 2 (SIMPLE_PINHOLE) calibration completed")

        return "WIP"
        # return {"success": True,
        #         "message": f"Calibration succeeded ({len(image_poses)} images)",
        #         "output_dir": output_dir,
        #         "camera_params": camera_params_out,
        #         "image_poses": image_poses,
        #         "num_registered_images": len(image_poses)}
    except Exception as e:
        return {"success": False, "message": f"Exception: {e}", "output_dir": output_dir}
