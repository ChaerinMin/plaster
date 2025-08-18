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
            # args.source = "~/data/brics/non-pii/brics-studio"
            # day = "2025-03-28" # brics-studio, multisequence000001
            # ms["name"] = "multisequence000001"
            args.source = "/oscar/data/ssrinath/brics/non-pii/brics-universe"
            day = "2025-05-14" # brics-universe, multisequence000003
            ms["name"] = "multisequence000003"

            print(f"Processing multisequence: {ms['name']}")
            dataloader = primer.Primer(args.source, day, ms["name"])
            data = dataloader.get_overlapping(lookup_thresh_ms=20)
            # Perform camera calibration (best-effort) for this multisequence
            # Get all the frames from data
            # SIMPLE_PINHOLE, PINHOLE: Use these camera models, if your images are undistorted a priori. These use one and two focal length parameters, respectively. Note that even in the case of undistorted images, COLMAP could try to improve the intrinsics with a more complex camera model.
            # SIMPLE_RADIAL, RADIAL: This should be the camera model of choice, if the intrinsics are unknown and every image has a different camera calibration, e.g., in the case of Internet photos. Both models are simplified versions of the OPENCV model only modeling radial distortion effects with one and two parameters, respectively.

            # OPENCV, FULL_OPENCV: Use these camera models, if you know the calibration parameters a priori. You can also try to let COLMAP estimate the parameters, if you share the intrinsics for multiple images. Note that the automatic estimation of parameters will most likely fail, if every image has a separate set of intrinsic parameters.

            # SIMPLE_RADIAL_FISHEYE, RADIAL_FISHEYE, OPENCV_FISHEYE, FOV, THIN_PRISM_FISHEYE, RAD_TAN_THIN_PRISM_FISHEYE: Use these camera models for fisheye lenses and note that all other models are not really capable of modeling the distortion effects of fisheye lenses. The FOV model is used by Google Project Tango (make sure to not initialize omega to zero).
            frame_data = [m.get("frame") for m in data["members"]]
            calib_dir = os.path.join(args.source, day, ms["name"])
            calib_res = calibrate_camera_from_primer(
                frame_data=frame_data,
                output_dir=calib_dir,
                camera_model="OPENCV_FISHEYE",
                clear_previous=False,
            )
            print(f"Calibration: {calib_res.get('message')}")