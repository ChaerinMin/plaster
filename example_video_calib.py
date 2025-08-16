"""Example script: calibrate camera intrinsics/extrinsics from uniformly
sampled video frames using Primer.video's FastVideoLoader.

Usage:
	python example_video_calib.py --video /path/to/video.mp4 --output ./calib_out --num-frames 12

Assumes dependencies (primer, pycolmap, pillow, torch, numpy) are installed.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List
import numpy as np

from primer.video import FastVideoLoader  # type: ignore
from calibration import calibrate_camera_from_primer

def _load_frames(video_path: str, num_frames: int, verbose: bool = False) -> List[Dict[str, Any]]:
	"""Load and return a list of frame dicts with keys: id, image.

	Uses FastVideoLoader + next() to iterate. Attempts to get total frames
	via attributes (num_frames / __len__). Falls back to sequential read.
	"""
	if not os.path.isfile(video_path):
		raise FileNotFoundError(f"Video file not found: {video_path}")

	loader = FastVideoLoader(video_path)
	total = loader.frame_count

	# Compute equally spaced indices
	if num_frames >= total:
		sample_indices = set(range(total))
	else:
		sample_indices = set(np.linspace(0, total - 1, num_frames, dtype=int).tolist())
	if verbose:
		print(f"Sampling {len(sample_indices)} frames out of {total}")

	frames: List[Dict[str, Any]] = []
	for idx in range(total):
		frame = loader.get_frame(idx)
		if idx in sample_indices:
			frames.append(frame)
		if len(frames) >= len(sample_indices):
			break
	return frames


def main():  # pragma: no cover - CLI entry
	parser = argparse.ArgumentParser(description="Calibrate camera from a subset of video frames")
	parser.add_argument("--video", required=True, help="Path to input video file")
	parser.add_argument("--output", required=True, help="Output directory for calibration artifacts")
	parser.add_argument("--num-frames", type=int, default=12, help="Number of equally spaced frames to sample")
	parser.add_argument("--camera-model", default="OPENCV", help="COLMAP camera model (default: OPENCV)")
	parser.add_argument("--min-images", type=int, default=5, help="Minimum images required for calibration")
	parser.add_argument("--verbose", action="store_true", help="Verbose logging")
	args = parser.parse_args()

	if args.num_frames < args.min_images:
		print(f"Warning: --num-frames ({args.num_frames}) < --min-images ({args.min_images}); increasing sample count.")
		args.num_frames = args.min_images

	try:
		frames = _load_frames(args.video, args.num_frames, verbose=args.verbose)
	except Exception as e:
		print(f"Failed to load frames: {e}")
		sys.exit(2)

	if len(frames) < args.min_images:
		print(f"Only extracted {len(frames)} frames (need at least {args.min_images}). Aborting.")
		sys.exit(3)

	if args.verbose:
		print(f"Collected {len(frames)} frames. Starting calibration...")

	result = calibrate_camera_from_primer(
		frame_data=frames,
		output_dir=args.output,
		camera_model=args.camera_model,
		min_images=args.min_images,
		clear_previous=False,
		verbose=args.verbose,
	)

	print("--- Calibration Result ---")
	for k, v in result.items():
		if k in ("image_poses",) and not args.verbose:
			print(f"{k}: (hidden, use --verbose to show {len(v)} entries)")
		else:
			print(f"{k}: {v}")

	if not result.get("success"):
		sys.exit(4)


if __name__ == "__main__":  # pragma: no cover
	main()

