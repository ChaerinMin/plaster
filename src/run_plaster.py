import source
import argparse

parser = argparse.ArgumentParser(description="Run Plaster with specified source.")
parser.add_argument("-s", "--source", type=str, help="Path to the source directory")
parser.add_argument("-f", "--force-reserialize", action="store_true", help="Force reserialization of the source")
# Take an argument for time stamp units
parser.add_argument("-t", "--time-stamp-units", type=str, choices=["microseconds", "nanoseconds"], default="nanoseconds", help="Time stamp units")

if __name__ == "__main__":
    args = parser.parse_args()

    source_instance = source.Source(args.source, force_reserialize=args.force_reserialize, time_stamp_units=args.time_stamp_units)