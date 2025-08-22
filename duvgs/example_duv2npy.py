# Write a basic usage example. Use argparse
import argparse
from duvgs import decode_videos_to_npy_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode DUV format to NPY files")
    parser.add_argument("input_dir", help="Input directory containing DUV files")
    parser.add_argument("output_dir", help="Output directory for NPY files")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--crf", type=int, default=23, help="Constant Rate Factor")
    parser.add_argument("--codec", default="libx264", help="Video codec")
    parser.add_argument("--pix_fmt", default="yuv420p", help="Pixel format")
    args = parser.parse_args()

    decode_videos_to_npy_dir(
        args.input_dir,
        args.output_dir,
        fps=args.fps,
        crf=args.crf,
        codec=args.codec,
        pix_fmt=args.pix_fmt,
    )
