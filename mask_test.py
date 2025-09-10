
import argparse
import os
import json
import cv2
import numpy as np
import vggt_colmap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask Generation Test Script")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to the source directory containing plaster.json")
    parser.add_argument("--thres-percent", type=float, default=50.0, help="Threshold value for VGGT-based filtering in percent.")
    
    args = parser.parse_args()
    
    predictions, image_names = vggt_colmap.run_model(args.input_dir)
    
    conf_mask = None
    # if 'depth_conf' in predictions:
    #     threshold_val = np.percentile(predictions['depth_conf'], conf_thres_percent)
    #     print('Threshold with value: ', threshold_val)
    #     print('Min and Max of depth conf: ', np.min(predictions['depth_conf']), np.max(predictions['depth_conf']))
    #     conf_mask = predictions['depth_conf'] >= threshold_val
    if 'depth' in predictions:
        threshold_val = np.percentile(predictions['depth'], args.thres_percent)
        print('Threshold with value: ', threshold_val)
        print('Min and Max of depth: ', np.min(predictions['depth']), np.max(predictions['depth']))
        conf_mask = predictions['depth'] < threshold_val
    
    os.makedirs(os.path.join(args.input_dir, "masks"), exist_ok=True)
    print('Depth map shape: ', predictions['depth'].shape)
    image_shape = cv2.imread(image_names[0]).shape
    print('Image shape: ', image_shape)
    for i in range(predictions['depth'].shape[0]):
        mask = conf_mask[i].astype(np.uint8) * 255
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_CUBIC)
        basename = os.path.basename(image_names[i])
        cv2.imwrite(os.path.join(args.input_dir, "masks", basename), mask)
        
    os.makedirs(os.path.join(args.input_dir, "images_masked"), exist_ok=True)
    for i in range(predictions['depth'].shape[0]):
        mask = conf_mask[i].astype(np.uint8) * 255
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_CUBIC)
        image = cv2.imread(image_names[i])
        image = cv2.bitwise_and(image, image, mask=mask)
        basename = os.path.basename(image_names[i])
        cv2.imwrite(os.path.join(args.input_dir, "images_masked", basename), image)
    print(f"Masks and masked images saved in {os.path.join(args.input_dir, 'masks')} and {os.path.join(args.input_dir, 'images_masked')}")