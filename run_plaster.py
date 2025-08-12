import source
import argparse
from time_cache import TimeCache
import os

parser = argparse.ArgumentParser(description="Run Plaster with specified source.")
parser.add_argument("-s", "--source", type=str, help="Path to the source directory")
parser.add_argument("-f", "--force-reserialize", action="store_true", help="Force reserialization of the source")

if __name__ == "__main__":
    args = parser.parse_args()
    if not args.source:
        print("Source directory is required. Use -s or --source to specify it.")
        parser.print_help()
        exit(1)

    source_instance = source.Source(args.source, force_reserialize=args.force_reserialize)

    # Recursively find all sensor directories in source-->day-->sensor. Then run time_cache
    day_dirs = [os.path.join(day.source_path, day.date) for day in source_instance.days]
    for (day_dir, day) in zip(day_dirs, source_instance.days):
        sensor_dirs = [os.path.join(day_dir, sensor.path) for sensor in day.sensors]
        for sensor_dir in sensor_dirs:
            print(sensor_dir)
            # time_cache_instance = TimeCache(sensor_dir, force_recompute=args.force_reserialize)
            # time_cache_instance.process()
