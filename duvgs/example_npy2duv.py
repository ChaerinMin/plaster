# Write a basic usage example. Use argparse
import argparse
from duvgs import encode_npy_dir_to_videos

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode NPY files to H.264 videos")
    parser.add_argument("input_dir", help="Input directory containing .npy files")
    parser.add_argument("output_path", help="Output .tar[.gz|.bz2|.xz] file or directory for videos")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--crf", type=int, default=23, help="Constant Rate Factor (0=lossless)")
    parser.add_argument("--codec", default="libx264", help="Video codec")
    parser.add_argument("--pix_fmt", default="rgb24", help="Pixel format (rgb24 recommended for quantized)")
    parser.add_argument("--quantize", action="store_true", help="Quantize floats to 8-bit per channel before encoding")
    parser.add_argument("--no-quantize", dest="quantize", action="store_false")
    parser.set_defaults(quantize=None)  # None => auto: quantize when crf>0
    parser.add_argument("--quant_bits", type=int, default=8, help="Quantization bits (currently only 8 supported)")
    args = parser.parse_args()

    encode_npy_dir_to_videos(
        args.input_dir,
    args.output_path,
        fps=args.fps,
        crf=args.crf,
        codec=args.codec,
    pix_fmt=args.pix_fmt,
    quantize=args.quantize,
    quant_bits=args.quant_bits,
    )
