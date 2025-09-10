
import argparse
import os
import json
import cv2
import numpy as np
import vggt_colmap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask Generation Test Script")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to the source directory containing plaster.json")
    parser.add_argument("--conf-thres-percent", type=float, default=60.0, help="Confidence threshold value for depth filtering in percent.")
    
    args = parser.parse_args()
    
    # _, _, depth_map, depth_conf, conf_mask, points_3d = vggt_colmap.run_vggt_custom(args.input_dir, conf_thres_percent=args.conf_thres_percent)
    _, _, depth_map, depth_conf, conf_mask, points_3d, image_names = vggt_colmap.run_model(args.input_dir, conf_thres_percent=args.conf_thres_percent)
    os.makedirs(os.path.join(args.input_dir, "masks"), exist_ok=True)
    print('Depth map shape: ', depth_map.shape)
    image_shape = cv2.imread(image_names[0]).shape
    print('Image shape: ', image_shape)
    for i in range(depth_map.shape[0]):
        mask = conf_mask[i].astype(np.uint8) * 255
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_CUBIC)
        basename = os.path.basename(image_names[i])
        cv2.imwrite(os.path.join(args.input_dir, "masks", basename), mask)