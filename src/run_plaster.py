import source
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Plaster with specified source.")
    parser.add_argument("source", type=str, help="Path to the source directory")
    args = parser.parse_args()
    source_instance = source.Source(args.source)
    print(f"Source Name: {source_instance.name}")
    print(f"Source Path: {source_instance.path}")

    # Assuming the directory structure is set up correctly, this will compute the days
    for day in source_instance.days:
        print(f"Day: {day.get_date()}")