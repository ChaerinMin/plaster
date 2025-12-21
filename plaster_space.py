import source
import argparse
from time_cache import TimeCache
import os
import json
import primer
import shutil
from datetime import datetime
from calibration import calibrate_camera_from_primer
import traceback
import sys
import pycolmap

parser = argparse.ArgumentParser(description="Run Plaster with specified source.")
parser.add_argument("-s", "--source", type=str, help="Path to the source directory", required=True)
parser.add_argument("-d", "--day", type=str, help="If looking for a specific day.", required=False, default=None)
parser.add_argument("--ms", type=str, help="If looking for a specific multisequence.", required=False, default=None)
parser.add_argument("--time-thresh", type=float, default=20, help="Time sync threshold in ms.")
parser.add_argument("-f", "--force-reserialize", action="store_true", help="Force reserialization of the source")
parser.add_argument("--max-features", type=int, default=15000, help="Maximum number of features to detect per image.")
parser.add_argument("--min-num-matches", type=int, default=15, help="Minimum number of matches required to consider a reconstruction valid.")
parser.add_argument("--conf-thres-percent", type=float, default=50.0, help="Confidence threshold value for depth filtering in percent.")
parser.add_argument("--wb-temp", type=int, default=4600, help="White balance temperature for image harmonization.")

# vggt
parser.add_argument("--scene_dir", type=str, required=False, help="Directory containing the scene images", default=None)
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--use_ba", action="store_true", default=False, help="Use BA for reconstruction")

if __name__ == "__main__":
    args = parser.parse_args()
    if not args.source:
        parser.print_help()
        exit(1)
        
    if os.path.exists(args.source) == False:
        print(f"Source directory not found: {args.source}")
        exit(1)
        
    if args.day is not None:
        if not os.path.exists(os.path.join(args.source, args.day)):
            print(f"Day directory not found: {os.path.join(args.source, args.day)}")
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
    for day in source_plaster["days"]:
        if args.day and day != args.day:
            continue
        print(f"Calibrating multisequences in day: {day}")

        if day == today:
            double_force_reserialize = True
            print(f"Today's date ({today}) found in source plaster. Force reserialize is set to {double_force_reserialize}.")
            
        day_plaster = json.load(open(os.path.join(args.source, day, "plaster.json"), 'r'))
        for ms in day_plaster["multisequences"]:
            if args.ms and ms["name"] != args.ms:
                continue
            calib_dir = os.path.join(args.source, day, ms["name"], "calib")
            if os.path.exists(calib_dir) and double_force_reserialize:
                print(f"Removing existing calibration directory: {calib_dir}")
                shutil.rmtree(calib_dir)
                
            if os.path.exists(calib_dir) and not double_force_reserialize:
                print(f"Calibration directory already exists: {calib_dir}. Skipping calibration for this multisequence.")
                continue

            try:
                print(f"Processing multisequence: {ms['name']}")
                dataloader = primer.Primer(args.source, day, ms["name"])
                data = dataloader.get_overlapping(lookup_thresh_ms=args.time_thresh, wb_temp=args.wb_temp, is_harmonize=True)

                frame_data = [
                    {"id": m["name"], "image": m["frame"]}
                    for m in data["members"]
                    if m.get("frame") is not None
                ]

                calib_res = calibrate_camera_from_primer(
                    frames=frame_data,
                    output_dir=calib_dir,
                    clear_previous=double_force_reserialize,
                    args=args,
                    stage1_camera_model = "OPENCV",
                    stage1_camera_mode = pycolmap.CameraMode.SINGLE,
                    stage2_camera_model = "PINHOLE",
                    stage2_camera_mode = pycolmap.CameraMode.SINGLE,
                    stage3_camera_model = "PINHOLE",
                    stage3_camera_mode = pycolmap.CameraMode.SINGLE,
                )
                print(f"Calibration: {calib_res}")
            except Exception as e:
                # Verbose logging with traceback and line numbers
                exc_type, exc_value, exc_tb = sys.exc_info()
                tb_list = traceback.extract_tb(exc_tb)
                formatted_tb = ''.join(traceback.format_list(tb_list))
                final_frame = tb_list[-1] if tb_list else None
                if final_frame:
                    print(
                        f"Error processing multisequence '{ms.get('name','UNKNOWN')}': {exc_type.__name__}: {e}\n"
                        f"  At {final_frame.filename}:{final_frame.lineno} in {final_frame.name}\n"
                        f"  Code: {final_frame.line}"
                    )
                else:
                    print(f"Error processing multisequence '{ms.get('name','UNKNOWN')}': {exc_type.__name__}: {e}")
                print("Full traceback (most recent call last):")
                print(formatted_tb.rstrip())
                continue