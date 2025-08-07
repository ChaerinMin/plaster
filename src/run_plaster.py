import source
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Plaster with specified source.")
    parser.add_argument("source", type=str, help="Path to the source directory")
    args = parser.parse_args()
    source_instance = source.Source(args.source, force_reserialize=True)