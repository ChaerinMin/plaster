import source
import argparse
from time_cache import TimeCache
import os
import primer

parser = argparse.ArgumentParser(description="Run Plaster with specified source.")
parser.add_argument("-s", "--source", type=str, help="Path to the source directory")
parser.add_argument("-f", "--force-reserialize", action="store_true", help="Force reserialization of the source")

if __name__ == "__main__":
    args = parser.parse_args()
    if not args.source:
        parser.print_help()
        exit(1)

    # First, cache the source directory and plaster.json
    source_instance = source.Source(args.source, force_reserialize=args.force_reserialize)

    # Time calibration: Recursively find all sensor directories in source-->day-->sensor. Then run time_cache
    day_dirs = [os.path.join(day.source_path, day.date) for day in source_instance.days]
    for (day_dir, day) in zip(day_dirs, source_instance.days):
        sensor_dirs = [os.path.join(day_dir, sensor.path) for sensor in day.sensors]
        for sensor_dir in sensor_dirs:
            try:
                print(f"Initializing TimeCache for sensor directory: {sensor_dir}")
                time_cache_instance = TimeCache(sensor_dir, force_recompute=args.force_reserialize)
            except Exception as e:
                print(f"Error initializing TimeCache for {sensor_dir}: {e}. Are you sure the environment is activated?")

    # Now let's use primer to get the data for spatial sensor calibration
    for day in source_instance.days:
        print(f"Calibrating multisequences in day: {day.date}")
        # Parse JSON in day directory to get multisequence names
        
        for ms in day.multisequences:
            print(f"Processing multisequence: {ms['name']}")
            dataloader = primer.Primer(source_instance.path, day.date, ms["name"])
            data = dataloader.get_overlapping(lookup_thresh_ms=20)
            print(data)