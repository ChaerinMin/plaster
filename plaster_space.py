import source
import argparse
from time_cache import TimeCache
import os
import json
import primer
import shutil
from datetime import datetime
from calibration import calibrate_camera_from_primer

parser = argparse.ArgumentParser(description="Run Plaster with specified source.")
parser.add_argument("-s", "--source", type=str, help="Path to the source directory")
parser.add_argument("-f", "--force-reserialize", action="store_true", help="Force reserialization of the source")
parser.add_argument("--run-vggt-stage3", action="store_true", help="Run VGGT Stage 3 calibration if VGGT is available.")

# VGGT COLMAP arguments. Used only if VGGT is available. Taken from https://raw.githubusercontent.com/facebookresearch/vggt/refs/heads/main/demo_colmap.py
vg_group = parser.add_argument_group("VGGT COLMAP")
vg_group.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
vg_group.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
vg_group.add_argument("--use_ba", action="store_true", default=False, help="Use BA for reconstruction")
######### BA parameters #########
vg_group.add_argument(
    "--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction"
)
vg_group.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
vg_group.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
vg_group.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
vg_group.add_argument("--query_frame_num", type=int, default=8, help="Number of frames to query")
vg_group.add_argument("--max_query_pts", type=int, default=4096, help="Maximum number of query points")
vg_group.add_argument(
    "--fine_tracking", action="store_true", default=True, help="Use fine tracking (slower but more accurate)"
)
vg_group.add_argument(
    "--conf_thres_value", type=float, default=5.0, help="Confidence threshold value for depth filtering (wo BA)"
)

if __name__ == "__main__":
    args = parser.parse_args()
    if not args.source:
        parser.print_help()
        exit(1)
        
    # First, cache the source directory and plaster.json
    source_plaster_path = os.path.join(args.source, "plaster.json")
    if not os.path.exists(source_plaster_path):
        print(f"Plaster file not found: {source_plaster_path}")
        exit(1)
    source_plaster = json.load(open(source_plaster_path, 'r'))

    # Now let's use primer to get the data for spatial sensor calibration
    double_force_reserialize = args.force_reserialize
    today = datetime.now().strftime('%Y-%m-%d') # Get today's date in YYYY-MM-DD format
    # # DEBUG
    # if True:
    #     if True:
    #         ms = dict()
    #         # # Baby dancing multisequence
    #         # # args.source = "/oscar/data/ssrinath//brics/non-pii/brics-studio"
    #         # args.source = "/mnt/brics-studio"
    #         # day = "2025-03-28" # brics-studio, multisequence000001
    #         # ms["name"] = "multisequence000001"
    #         # # CS Lawn multisequence
    #         # # args.source = "/oscar/data/ssrinath/brics/non-pii/brics-universe"
    #         # args.source = "/mnt/brics-universe"
    #         # day = "2025-05-14" # brics-universe, multisequence000003
    #         # ms["name"] = "multisequence000003"
    #         # # 191 Medway
    #         # args.source = "/oscar/data/ssrinath/brics/non-pii/brics-universe"
    #         # day = "2025-05-11" # brics-universe, multisequence000003
    #         # ms["name"] = "multisequence000001"
    #         # Basketball pass multisequence
    #         # args.source = "/oscar/data/ssrinath//brics/non-pii/brics-studio"
    #         args.source = "/mnt/brics-studio"
    #         day = "2025-04-23" # brics-studio, multisequence000001
    #         ms["name"] = "multisequence000001"
    # # END DEBUG
    # PRODUCTION
    for day in source_plaster["days"]:
        print(f"Calibrating multisequences in day: {day}")

        if day == today:
            double_force_reserialize = True
            print(f"Today's date ({today}) found in source plaster. Force reserialize is set to {double_force_reserialize}.")
            
        day_plaster = json.load(open(os.path.join(args.source, day, "plaster.json"), 'r'))
        for ms in day_plaster["multisequences"]:
    # END PRODUCTION
            calib_dir = os.path.join(args.source, day, ms["name"], "calib")
            if os.path.exists(calib_dir) and double_force_reserialize:
                print(f"Removing existing calibration directory: {calib_dir}")
                shutil.rmtree(calib_dir)
                
            if os.path.exists(calib_dir) and not double_force_reserialize:
                print(f"Calibration directory already exists: {calib_dir}. Skipping calibration for this multisequence.")
                continue

            print(f"Processing multisequence: {ms['name']}")
            dataloader = primer.Primer(args.source, day, ms["name"])
            data = dataloader.get_overlapping(lookup_thresh_ms=20)

            frame_data = [{"id": m["name"], "image": m["frame"]} for m in data["members"]]

            calib_res = calibrate_camera_from_primer(
                frames=frame_data,
                output_dir=calib_dir,
                clear_previous=double_force_reserialize,
                args=args,
            )
            print(f"Calibration: {calib_res}")