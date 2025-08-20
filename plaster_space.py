import source
import argparse
from time_cache import TimeCache
import os
import json
import primer
from calibration import calibrate_camera_from_primer

parser = argparse.ArgumentParser(description="Run Plaster with specified source.")
parser.add_argument("-s", "--source", type=str, help="Path to the source directory")
parser.add_argument("-f", "--force-reserialize", action="store_true", help="Force reserialization of the source")

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

    # # Now let's use primer to get the data for spatial sensor calibration
    # for day in source_plaster["days"]:
    #     print(f"Calibrating multisequences in day: {day}")
    #     day_plaster = json.load(open(os.path.join(args.source, day, "plaster.json"), 'r'))

    #     for ms in day_plaster["multisequences"]:
    # DEBUG
    if True:
        if True:
            ms = dict()
            # Baby dancing multisequence
            args.source = "/oscar/data/ssrinath//brics/non-pii/brics-studio"
            day = "2025-03-28" # brics-studio, multisequence000001
            ms["name"] = "multisequence000001"
            # # CS Lawn multisequence
            # args.source = "/oscar/data/ssrinath/brics/non-pii/brics-universe"
            # day = "2025-05-14" # brics-universe, multisequence000003
            # ms["name"] = "multisequence000003"
            # # 191 Medway
            # args.source = "/oscar/data/ssrinath/brics/non-pii/brics-universe"
            # day = "2025-05-11" # brics-universe, multisequence000003
            # ms["name"] = "multisequence000001"

            print(f"Processing multisequence: {ms['name']}")
            dataloader = primer.Primer(args.source, day, ms["name"])
            data = dataloader.get_overlapping(lookup_thresh_ms=20)

            frame_data = [{"id": m["name"], "image": m["frame"]} for m in data["members"]]

            # frame_data = [{"id": m["name"], "image": m["frame"]} for m in data["members"][:10]]
            # print(f'DEBUG with {frame_data.__len__()} images.')

            calib_dir = os.path.join(args.source, day, ms["name"], "calib")
            calib_res = calibrate_camera_from_primer(
                frames=frame_data,
                output_dir=calib_dir,
                clear_previous=args.force_reserialize,
            )
            print(f"Calibration: {calib_res}")