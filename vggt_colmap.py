# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path
import trimesh
import pycolmap
import cv2

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square, load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.dependency.projection import project_3D_points_np
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track


# TODO: add support for masks
# TODO: add iterative BA
# TODO: add support for radial distortion, which needs extra_params
# TODO: test with more cases
# TODO: test different camera types

VGGT_WEIGHTS_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
# VGGT Commercial
# VGGT_WEIGHTS_URL = "https://huggingface.co/facebook/VGGT-1B-Commercial/blob/main/vggt_1B_commercial.pt"

def run_VGGT(model, images, dtype, resolution=518):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    print(f"Setting seed as: {seed}")
    
def set_device_and_dtype():
    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    return device, dtype
    
# -------------------------------------------------------------------------
# 1) Core model inference
# -------------------------------------------------------------------------
def run_model(target_dir, seed=42) -> dict:
    """
    Run the VGGT model on images in the 'target_dir/images' folder and return predictions.
    """
    set_seed(seed)
    device, dtype = set_device_and_dtype()

    # Run VGGT for camera and depth estimation
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(VGGT_WEIGHTS_URL))
    model.eval()
    model = model.to(device)
    print(f"VGGT model loaded")
    
    print(f"Processing images from {target_dir}")
    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)
    # images_old, original_coords = load_and_preprocess_images_square(image_names, img_load_resolution)
    print(f"Preprocessed images shape: {images.shape}")

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    predictions['pose_enc_list'] = None # remove pose_enc_list

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    # Clean up
    torch.cuda.empty_cache()
    
    return predictions, image_names

def run_vggt_calibration(args):
    # Print configuration
    print("Arguments:", vars(args))
    set_seed(args.seed)
    device, dtype = set_device_and_dtype()  

    # Run VGGT for camera and depth estimation
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(VGGT_WEIGHTS_URL))
    model.eval()
    model = model.to(device)
    print(f"Model loaded")

    # Get image paths and preprocess them
    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list]
    
    # Load images and original coordinates
    # Load Image in 1024, while running VGGT with 518
    vggt_fixed_resolution = 518
    img_load_resolution = 1024

    # images_old, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    # images = load_and_preprocess_images(image_path_list).to(device)
    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False)
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
            
    # # unload the model to save GPU memory
    # model.cpu()
    # torch.cuda.empty_cache()

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = predictions["depth"].squeeze(0).cpu().numpy()
    depth_conf = predictions["depth_conf"].squeeze(0).cpu().numpy()

    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    if args.use_ba:
        print('WARNING: BA might run out of memory. Try without --use_ba if that happens.')
        image_size = np.array(images.shape[-2:])
        scale = img_load_resolution / vggt_fixed_resolution
        shared_camera = args.shared_camera

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                # Predicting Tracks
                # Using VGGSfM tracker instead of VGGT tracker for efficiency
                # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
                # Will be fixed in VGGT v2

                # You can also change the pred_tracks to tracks from any other methods
                # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
                pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                    images,
                    conf=depth_conf,
                    points_3d=points_3d,
                    masks=None,
                    max_query_pts=args.max_query_pts,
                    query_frame_num=args.query_frame_num,
                    keypoint_extractor="aliked+sp",
                    fine_tracking=args.fine_tracking,
                )
                
                torch.cuda.empty_cache()
            
        # rescale the intrinsic matrix from 518 to 1024
        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > args.vis_thresh

        # TODO: radial distortion, iterative BA, masks
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=args.max_reproj_error,
            shared_camera=shared_camera,
            camera_type=args.camera_type,
            points_rgb=points_rgb,
        )

        if reconstruction is None:
            raise ValueError("No reconstruction can be built with BA")

        # Bundle Adjustment
        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)

        reconstruction_resolution = img_load_resolution
    else:
        conf_threshold_val = np.percentile(depth_conf, args.conf_thres_percent)
        print('Threshold with value: ', conf_threshold_val)
        print('Min and Max of depth conf: ', np.min(depth_conf), np.max(depth_conf))
        
        depth_threshold_val = np.percentile(depth_map, args.depth_thres_percent)
        print('Depth Threshold with value (NOT YET IMPLEMENTED): ', depth_threshold_val)
        print('Min and Max of depth: ', np.min(depth_map), np.max(depth_map))

        max_points_for_colmap = 200000  # randomly sample 3D points
        shared_camera = False  # in the feedforward manner, we do not support shared camera
        camera_type = "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera

        image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
        num_frames, height, width, _ = points_3d.shape

        points_rgb = F.interpolate(
            images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        # (S, H, W, 3), with x, y coordinates and frame indices
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        conf_mask = depth_conf >= conf_threshold_val
        # at most writing 100000 3d points to colmap reconstruction object
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        print("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
        )

        reconstruction_resolution = vggt_fixed_resolution

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )   

    print(f"Saving reconstruction to {args.scene_dir}/sparse")
    sparse_reconstruction_dir = os.path.join(args.scene_dir, "sparse")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(args.scene_dir, "sparse/points.ply"))
    
    print(f"Saving masked images to {args.scene_dir}/images_masked")
    os.makedirs(os.path.join(args.scene_dir, "images_masked"), exist_ok=True)

    print(f"images.shape: {images.shape}, points_3d.shape: {points_3d.shape}")
    
    scale = img_load_resolution / vggt_fixed_resolution
    image_shape = cv2.imread(image_path_list[0]).shape
    
    for i in range(images.shape[0]):
        mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)

        # Filter invalid/nans
        valid_pts = np.isfinite(points_3d).all(axis=1)
        pts3d_img = points_3d[valid_pts]

        if pts3d_img.size != 0:
            # Prepare batched extrinsics/intrinsics for NumPy projector
            extr_b = extrinsic[i][None, ...] # (1,3,4)
            pyimage = reconstruction.images[i+1] # pycolmap image ids start from 1
            pycamera = reconstruction.cameras[pyimage.camera_id]
            intr_b = make_intrinsic_from_pycamera(pycamera)
            # print(f"Image {i}: extr_b shape: {extr_b.shape}, intr_b shape: {intr_b.shape}, pts3d_img shape: {pts3d_img.shape}")
            
            pts2d_t, pts_cam_t = project_3D_points_np(
                pts3d_img, extr_b, intr_b, default=0.0, only_points_cam=False
            )

            # Extract and filter valid projected points
            pts2d = pts2d_t[0]             # (N,2) as np.ndarray
            z_cam = pts_cam_t[0, 2]        # (N,) as np.ndarray
            x_f = pts2d[:, 0]
            y_f = pts2d[:, 1]
            valid = (
                np.isfinite(x_f) & np.isfinite(y_f) & np.isfinite(z_cam) & (z_cam > 0)
            )
            if np.any(valid):
                x = np.rint(x_f[valid]).astype(np.int32)
                y = np.rint(y_f[valid]).astype(np.int32)
                in_bounds = (x >= 0) & (y >= 0) & (x < image_shape[1]) & (y < image_shape[0])
                if np.any(in_bounds):
                    mask[y[in_bounds], x[in_bounds]] = 255
        else:
            print(f"Warning: No valid 3D points for image {i}, saving empty mask.")

        # # Load original image (BGR) and apply mask
        in_path = image_path_list[i]        
        # We are faking point splatting by dilation
        kernel_size = 11
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
        # Downsize and upsize again to remove small holes
        mask = cv2.resize(mask, (image_shape[1]//2, image_shape[0]//2), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)

        img_bgr = cv2.imread(in_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"Warning: failed to read image {in_path}; skipping.")
            continue
        # Create BGRA image with mask in alpha channel
        img_bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
        img_bgra[:, :, 3] = mask
        out_path = os.path.join(args.scene_dir, "images_masked", os.path.basename(in_path) + '.png')
        ok = cv2.imwrite(out_path, img_bgra)
        # ok = cv2.imwrite(out_path, mask)
        if not ok:
            print(f"Warning: failed to write masked image {out_path}")

    return True

def rename_colmap_recons(
    reconstruction, image_paths):

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

    return reconstruction

def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]
            # print(f"Rescaled camera {pyimageid} with ratio {resize_ratio}, new size: {pycamera.width}x{pycamera.height}")

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction


def make_intrinsic_from_pycamera(pycamera):
    if pycamera.model in [pycolmap.CameraModelId.SIMPLE_PINHOLE, pycolmap.CameraModelId.SIMPLE_RADIAL]:
        fx = fy = pycamera.params[0]
        cx = pycamera.params[1]
        cy = pycamera.params[2]
    elif pycamera.model in [pycolmap.CameraModelId.PINHOLE, pycolmap.CameraModelId.RADIAL]:
        fx = pycamera.params[0]
        fy = pycamera.params[1]
        cx = pycamera.params[2]
        cy = pycamera.params[3]
    else:
        raise NotImplementedError(f"Camera model {pycamera.model} not supported yet")
    intr_b = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])[None, ...]  # (1,3,3)
    return intr_b