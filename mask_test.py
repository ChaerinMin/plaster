
import argparse
import os
import json
import cv2
import numpy as np
import vggt_colmap

parser = argparse.ArgumentParser(description="Mask Generation Test Script")

parser.add_argument("--conf-thres-percent", type=float, default=50.0, help="Confidence threshold value for depth filtering in percent.")


# VGGT COLMAP arguments. Used only if VGGT is available. Taken from https://raw.githubusercontent.com/facebookresearch/vggt/refs/heads/main/demo_colmap.py
vg_group = parser.add_argument_group("VGGT COLMAP")
vg_group.add_argument("--scene_dir", type=str, required=False, help="Directory containing the scene images", default=None)
vg_group.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
vg_group.add_argument("--use_ba", action="store_true", default=False, help="Use BA for reconstruction")
######### BA parameters #########
vg_group.add_argument(
    "--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction"
)
vg_group.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
vg_group.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
vg_group.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
vg_group.add_argument("--query_frame_num", type=int, default=2, help="Number of frames to query")
vg_group.add_argument("--max_query_pts", type=int, default=4096, help="Maximum number of query points")
vg_group.add_argument(
    "--fine_tracking", action="store_true", default=False, help="Use fine tracking (slower but more accurate)"
)
vg_group.add_argument("--max-3d-points", type=int, default=1000000, help="Maximum number of 3D points to use")
vg_group.add_argument("--dilation-kernel-size", type=int, default=13, help="Dilation kernel size for mask refinement")

if __name__ == "__main__":
    args = parser.parse_args()
    
    vggt_colmap.run_vggt_calibration(args)
    
    # predictions, image_names = vggt_colmap.run_model(args.input_dir)
    
    # conf_mask = None
    # if 'depth_conf' in predictions:
    #     threshold_val = np.percentile(predictions['depth_conf'], args.thres_percent)
    #     print('Threshold with value: ', threshold_val)
    #     print('Min and Max of depth conf: ', np.min(predictions['depth_conf']), np.max(predictions['depth_conf']))
    #     conf_mask = predictions['depth_conf'] >= threshold_val
    # # if 'depth' in predictions:
    # #     threshold_val = np.percentile(predictions['depth'], args.thres_percent)
    # #     print('Threshold with value: ', threshold_val)
    # #     print('Min and Max of depth: ', np.min(predictions['depth']), np.max(predictions['depth']))
    # #     conf_mask = predictions['depth'] < threshold_val
    
    # os.makedirs(os.path.join(args.input_dir, "masks"), exist_ok=True)
    # print('Depth map shape: ', predictions['depth'].shape)
    # image_shape = cv2.imread(image_names[0]).shape
    # print('Image shape: ', image_shape)
    # for i in range(predictions['depth'].shape[0]):
    #     mask = conf_mask[i].astype(np.uint8) * 255
    #     mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_CUBIC)
    #     basename = os.path.basename(image_names[i])
    #     cv2.imwrite(os.path.join(args.input_dir, "masks", basename), mask)
        
    # os.makedirs(os.path.join(args.input_dir, "images_masked"), exist_ok=True)
    # for i in range(predictions['depth'].shape[0]):
    #     mask = conf_mask[i].astype(np.uint8) * 255
    #     mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_CUBIC)
    #     image = cv2.imread(image_names[i])
    #     image = cv2.bitwise_and(image, image, mask=mask)
    #     basename = os.path.basename(image_names[i])
    #     cv2.imwrite(os.path.join(args.input_dir, "images_masked", basename), image)
    # print(f"Masks and masked images saved in {os.path.join(args.input_dir, 'masks')} and {os.path.join(args.input_dir, 'images_masked')}")