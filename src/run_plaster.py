import source
import argparse

parser = argparse.ArgumentParser(description="Run Plaster with specified source.")
parser.add_argument("-s", "--source", type=str, help="Path to the source directory")
parser.add_argument("-f", "--force-reserialize", action="store_true", help="Force reserialization of the source")
# Take an argument for time stamp units
parser.add_argument("-t", "--time-stamp-units", type=str, choices=["microseconds", "nanoseconds"], default="nanoseconds", help="Time stamp units")

TIME_MULTIPLIER = 1e9 # Nanoseconds by default
TIME_THRESHOLD_S = 5 # THRESHOLD in seconds

if __name__ == "__main__":
    args = parser.parse_args()
    if args.time_stamp_units == "microseconds":
        print("Using microseconds for time stamps.")
        TIME_MULTIPLIER = 1e6
    else:
        print("Using nanoseconds for time stamps.")
        TIME_MULTIPLIER = 1e9

    source_instance = source.Source(args.source, force_reserialize=args.force_reserialize)