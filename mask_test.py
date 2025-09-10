
import argparse
import os
import json
import cv2
import numpy as np
import vggt_colmap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask Generation Test Script")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to the source directory containing plaster.json")
    parser.add_argument("--conf-thres-value", type=float, default=5.0, help="Confidence threshold value for depth filtering")
    
    args = parser.parse_args()
    
    _, _, depth_map, depth_conf, conf_mask, points_3d = vggt_colmap.run_vggt_custom(args.input_dir, conf_thres_value=args.conf_thres_value)
    os.makedirs(os.path.join(args.input_dir, "masks"), exist_ok=True)
    print('Depth map shape: ', depth_map.shape)
    for i in range(depth_map.shape[0]):
        mask = conf_mask[i].astype(np.uint8) * 255
        # mask = cv2.resize(mask, (best_recon.images[i].width, best_recon.images[i].height), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(args.input_dir, "masks", f"mask_{i:03d}.png"), mask)