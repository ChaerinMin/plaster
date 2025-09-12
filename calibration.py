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
import numpy as np
import pycolmap
import glob
import json
from primer.helpers import undistort_images
import cv2
import argparse

# Check if VGGT is install for stage3 calibration
try:
    import vggt
    import vggt_colmap
    VGGT_FOUND = True
    print("VGGT module found. Stage 3 calibration is available.")
except ImportError:
    VGGT_FOUND = False
    print("VGGT module not found. Stage 3 calibration is unavailable.")

def _prepare_image_array(img: Any) -> Optional[np.ndarray]:
    """Normalize an input array-like into a uint8 numpy array acceptable by OpenCV.

    Accepts shapes:
      - H x W (grayscale)
      - H x W x C (C in 1,3,4)
      - C x H x W (C in 1,3,4)  (will transpose)

    Handles dtypes float32/float64 (assumed in range 0-1 or 0-255) and integer types.
    Returns None if the array cannot be interpreted as an image.
    """
    if img is None:
        return None

    if not isinstance(img, np.ndarray):
        return None
    if img.ndim == 0:
        return None

    # Remove extraneous batch dimension if present: (1, C, H, W) or (1, H, W, C)
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]

    # CHW -> HWC if needed
    if img.ndim == 3:
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
    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))

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

    # Drop alpha for JPEG compatibility when using OpenCV
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # Final sanity checks
    if img.ndim == 3 and img.shape[2] not in (3,):
        # Unexpected channel count
        return None
    if img.ndim not in (2, 3):
        return None
    return img


def _write_images(frames: List[Dict[str, Any]], image_dir: str) -> List[Tuple[int, str]]:
    # Write frames to files. frames can be tensor-like or ndarrays in various layouts.
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
            # OpenCV expects BGR input; convert from RGB if needed so the saved file displays with correct RGB colors.
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 and img.shape[2] == 3 else img
            success = cv2.imwrite(out_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not success:
                raise RuntimeError("cv2.imwrite returned False")
        except Exception as e:
            print(f"Failed to write image {frame_id}: {e} (original shape {getattr(raw_img, 'shape', None)})")
            continue
        if os.path.exists(out_path):
            written.append((frame_id, out_path))
    return written

def calibrate_camera_from_primer(frames: Any,
                                 output_dir: str,
                                 args: argparse.Namespace,
                                 clear_previous: bool = False,
                                 min_images: int = 5,
                                 stage1_camera_model: str = "OPENCV",
                                 stage1_camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.SINGLE,
                                 stage2_camera_model: str = "PINHOLE",
                                 stage2_camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.SINGLE,
                                 stage3_camera_model: str = "PINHOLE",
                                 stage3_camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.SINGLE,
                                 ) -> Dict[str, Any]:
    if len(frames) < min_images:
        return {"success": False, "message": f"Need >= {min_images} frames", "output_dir": output_dir}
    
    if clear_previous and os.path.isdir(output_dir):
        print(f"Clearing previous output directory: {output_dir}")
        shutil.rmtree(output_dir, ignore_errors=True)

    stage1_dir = os.path.join(output_dir, "stage1")
    stage2_dir = os.path.join(output_dir, "stage2")
    stage3_dir = os.path.join(output_dir, "stage3")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)
    if VGGT_FOUND:
        os.makedirs(stage3_dir, exist_ok=True)
    pycolmap.set_log_destination(os.path.join(output_dir, "calib.log"))

    frame_path_list = _write_images(frames, os.path.join(stage1_dir, "images"))
    if len(frame_path_list) < min_images:
        return {"success": False, "message": f"Only {len(frame_path_list)} valid images", "output_dir": output_dir}

    final_cam_params = {
                    "stage1_model": "UNKNOWN",
                    "stage1_camera_mode": "UNKNOWN",
                    "stage1_params": None,
                    "stage2_model": "UNKNOWN",
                    "stage2_camera_mode": "UNKNOWN",
                    "stage2_params": None,
                }
    if VGGT_FOUND:
        final_cam_params["stage3_model"] = "UNKNOWN"
        final_cam_params["stage3_camera_mode"] = "UNKNOWN"
        final_cam_params["stage3_params"] = None

    # We will follow a multi-stage strategy
    # Stage 1 assumes a single camera model for all cameras, extracts distortion parameters and undistorts the images
    # Stage 2 takes the undistorted images and refines the camera parameters using a PINHOLE model
    try:
        stage1_database_path = os.path.join(stage1_dir, "stage1.db")
        stage1_image_dir = os.path.join(stage1_dir, "images")
        stage1_recon_output_path = os.path.join(stage1_dir, "sparse")

        sift_options = pycolmap.SiftExtractionOptions()
        sift_options.max_num_features = args.max_features # Maximize number of features

        print(f"Stage 1 ({stage1_camera_model} and {str(stage1_camera_mode)}) calibration started")
        pycolmap.extract_features(database_path=stage1_database_path, image_path=stage1_image_dir, camera_mode=stage1_camera_mode, camera_model=stage1_camera_model, sift_options=sift_options)

        pycolmap.match_exhaustive(database_path=stage1_database_path)
        
        # Reconstruction
        incremental_options = pycolmap.IncrementalPipelineOptions()
        incremental_options.multiple_models = False # Avoid multiple models
        incremental_options.max_num_models = 1
        incremental_options.ba_global_function_tolerance = 0.000001
        incremental_options.min_num_matches = args.min_num_matches # See https://github.com/colmap/colmap/issues/1225
        incremental_options.min_model_size = 5
        stage1_reconstruction = pycolmap.incremental_mapping(
            database_path=stage1_database_path,
            image_path=stage1_image_dir,
            output_path=stage1_recon_output_path,
            options=incremental_options
        )
        if stage1_reconstruction is None or len(stage1_reconstruction) == 0:
            raise RuntimeError("Stage 1 reconstruction failed or returned no models.")
        
        print(f'Found multiple ({len(stage1_reconstruction)}) reconstructions:')
        best_recon = None # Pick one with the most reconstruction
        for ctr, _ in enumerate(stage1_reconstruction):
            recon = stage1_reconstruction[ctr]
            print(f" - {recon.num_frames()} frames")
            if best_recon is None or recon.num_frames() > best_recon.num_frames():
                best_recon = recon

        stage2_image_dir = os.path.join(stage2_dir, "images")
        os.makedirs(stage2_image_dir, exist_ok=True)
        if VGGT_FOUND and args.run_vggt_stage3:
            stage3_image_dir = os.path.join(stage3_dir, "images")
            os.makedirs(stage3_image_dir, exist_ok=True)

        final_cam_params["stage1_model"] = str(stage1_camera_model)
        final_cam_params["stage1_camera_mode"] = str(stage1_camera_mode)
        for cam in best_recon.cameras.values():
            final_cam_params["stage1_params"] = cam.params.tolist()

            # OpenCV undistort
            for id, dist_img_path in frame_path_list:
                img = cv2.imread(dist_img_path)
                print(f'Undistorting with {cam.params.tolist()}')
                undist_img = undistort_images(input_img=img, camera_params=cam.params.tolist(), camera_model=stage1_camera_model)
                # undist_img is from OpenCV pipeline (BGR); save directly for correct RGB display.
                cv2.imwrite(os.path.join(stage2_image_dir, f"{id}.jpg"), undist_img)
                if VGGT_FOUND and args.run_vggt_stage3:
                    # undist_img is from OpenCV pipeline (BGR); save directly for correct RGB display.
                    cv2.imwrite(os.path.join(stage3_image_dir, f"{id}.jpg"), undist_img)

            break # Since we are assuming only 1 set of distortion parameters for all cameras

        # # COLMAP undistort
        # best_recon.write(stage1_dir) # Write explicitly
        # undistort_options = pycolmap.UndistortCameraOptions()
        # undistort_options.max_image_size = 1920 # PARAM
        # pycolmap.undistort_images(output_path=stage2_image_dir, input_path=stage1_dir, image_path=stage1_image_dir, undistort_options=undistort_options)
        # # Re-structure undistorted_image_dir
        # file_names = os.listdir(os.path.join(stage2_image_dir, "images"))
        # for file_name in file_names:
        #     # print(f"Moving {file_name} to {stage2_image_dir}/")
        #     shutil.move(os.path.join(stage2_image_dir, "images", file_name), os.path.join(stage2_image_dir, file_name))
        # shutil.rmtree(os.path.join(stage2_image_dir, "images"), ignore_errors=True)
        # shutil.rmtree(os.path.join(stage2_image_dir, "sparse"), ignore_errors=True)
        # shutil.rmtree(os.path.join(stage2_image_dir, "stereo"), ignore_errors=True)
        # sh_files = glob.glob(os.path.join(stage2_image_dir, "run-*.sh"))
        # for sh_file in sh_files:
        #     shutil.rmtree(sh_file, ignore_errors=True)

        print(f"Stage 1 ({stage1_camera_model} and {str(stage1_camera_mode)}) calibration completed with {len(stage1_reconstruction)} reconstructions and {best_recon.num_frames()} images for the best reconstruction.")
    except Exception as e:
        print(f"Stage 1 ({stage1_camera_model} and {str(stage1_camera_mode)}) calibration failed: {e}. Not proceeding to Stage 2. Exiting.")
        return {"success": False, "message": f"Exception: {e}", "output_dir": output_dir}

    try:
        stage2_database_path = os.path.join(stage2_dir, "stage2.db")
        stage2_recon_output_path = os.path.join(stage2_dir, "sparse")

        sift_options = pycolmap.SiftExtractionOptions()
        sift_options.max_num_features = args.max_features # Maximize number of features
        # SIMPLE_PINHOLE, PINHOLE: Use these camera models, if your images are undistorted a priori. These use one and two focal length parameters, respectively. Note that even in the case of undistorted images, COLMAP could try to improve the intrinsics with a more complex camera model.
        # OPENCV, FULL_OPENCV: Use these camera models, if you know the calibration parameters a priori. You can also try to let COLMAP estimate the parameters, if you share the intrinsics for multiple images. Note that the automatic estimation of parameters will most likely fail, if every image has a separate set of intrinsic parameters.
        print(f"Stage 2 ({stage2_camera_model} and {str(stage2_camera_mode)}) calibration started")
        pycolmap.extract_features(database_path=stage2_database_path, image_path=stage2_image_dir, camera_mode=stage2_camera_mode, camera_model=stage2_camera_model, sift_options=sift_options)

        pycolmap.match_exhaustive(database_path=stage2_database_path)

        # Reconstruction
        incremental_options = pycolmap.IncrementalPipelineOptions()
        incremental_options.multiple_models = False # Avoid multiple models
        incremental_options.max_num_models = 1
        incremental_options.ba_global_function_tolerance = 0.000001
        incremental_options.min_num_matches = args.min_num_matches # See https://github.com/colmap/colmap/issues/1225
        incremental_options.min_model_size = 5        
        stage2_reconstruction = pycolmap.incremental_mapping(
            database_path=stage2_database_path,
            image_path=stage2_image_dir,
            output_path=stage2_recon_output_path,
            options=incremental_options
        )
        if stage2_reconstruction is None or len(stage2_reconstruction) == 0:
            raise RuntimeError("Stage 2 reconstruction failed or returned no models.")
        
        print(f'Found multiple ({len(stage2_reconstruction)}) reconstructions:')
        best_recon = None # Pick one with the most reconstruction
        for ctr, _ in enumerate(stage2_reconstruction):
            recon = stage2_reconstruction[ctr]
            print(f" - {recon.num_frames()} frames")
            if best_recon is None or recon.num_frames() > best_recon.num_frames():
                best_recon = recon

        best_recon.write(output_dir) # Write the best reconstruction to the top level directory

        final_cam_params["stage2_model"] = str(stage2_camera_model)
        final_cam_params["stage2_camera_mode"] = str(stage2_camera_mode)
        for cam in best_recon.cameras.values():
            final_cam_params["stage2_params"] = cam.params.tolist()
            break

        with open(os.path.join(output_dir, f"final_cam_params.json"), "w") as f:
            f.write(json.dumps(final_cam_params, indent=4))

        print(f"Stage 2 ({stage2_camera_model} and {str(stage2_camera_mode)}) calibration completed with {best_recon.num_frames()} images for the best reconstruction.")
        
        # # If VGGT is available, let's actually get some masks
        # if VGGT_FOUND:
        #     predictions, image_names = vggt_colmap.run_model(args.input_dir)

        #     image_shape = cv2.imread(image_names[0]).shape
        #     os.makedirs(os.path.join(stage2_dir, "masks"), exist_ok=True)
        #     # print('Depth map shape: ', depth_map.shape)
        #     for i in range(depth_map.shape[0]):
        #         mask = conf_mask[i].astype(np.uint8) * 255
        #         mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_CUBIC)
        #         basename = os.path.basename(images_path[i])
        #         cv2.imwrite(os.path.join(stage2_dir, "masks", basename), mask)
                
        #     print(f"VGGT depth map shape: {depth_map.shape}, Depth conf shape: {depth_conf.shape}, Points 3D shape: {points_3d.shape}")
            
    except Exception as e:
        print(f"Stage 2 ({stage2_camera_model} and {str(stage2_camera_mode)}) calibration failed: {e}. Not proceeding to Stage 3. Exiting.")
        return {"success": False, "message": f"Exception: {e}", "output_dir": output_dir}
        
    try:
        if(args.run_vggt_stage3 and VGGT_FOUND):
            print(f"VGGT Stage 3 ({stage3_camera_model} and {str(stage3_camera_mode)}) calibration started.")
            if args.scene_dir is None:
                args.scene_dir = stage3_dir
            vggt_colmap.demo_fn(args)

        return {"success": True,
                "message": f"Calibration succeeded with {best_recon.num_frames()} images",
                "output_dir": output_dir,
                # "camera_params": camera_params_out,
                # "image_poses": image_poses,
                "num_registered_images": best_recon.num_frames()}
    except Exception as e:
        return {"success": False, "message": f"Exception: {e}", "output_dir": output_dir}
