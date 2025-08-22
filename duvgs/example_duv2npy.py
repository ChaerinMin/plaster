# Write a basic usage example. Use argparse
import argparse
from duvgs import decode_videos_to_npy_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode DUV archive/dir to NPY files")
    parser.add_argument("input_path", help="Input tar file or directory containing DUV files")
    parser.add_argument("output_dir", help="Output directory for NPY files")
    args = parser.parse_args()

    decode_videos_to_npy_dir(
        args.input_path,
        args.output_dir
    )
