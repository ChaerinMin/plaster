import argparse
import os
import shutil
import tempfile
from typing import Tuple

import numpy as np

from duvgs import encode_npy_dir_to_videos, decode_videos_to_npy_dir


def compute_errors(orig_dir: str, recon_dir: str) -> Tuple[float, float, float]:
    orig_files = [f for f in os.listdir(orig_dir) if f.lower().endswith(".npy")]
    recon_files = [f for f in os.listdir(recon_dir) if f.lower().endswith(".npy")]
    orig_files.sort()
    recon_files.sort()
    if orig_files != recon_files:
        raise RuntimeError("Original and reconstructed directories have different file sets")

    max_abs = 0.0
    mae_sum = 0.0
    count = 0
    for fname in orig_files:
        a = np.load(os.path.join(orig_dir, fname))
        b = np.load(os.path.join(recon_dir, fname))
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError(f"Mismatch for {fname}: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
        diff = np.abs(a - b)
        max_abs = max(max_abs, float(diff.max()))
        mae_sum += float(diff.mean())
        count += 1
    mae = mae_sum / max(count, 1)
    # Relative MAE normalized by dynamic range (per-file mean). Avoid div by zero.
    # Here we approximate using overall range from first file.
    if count > 0:
        a0 = np.load(os.path.join(orig_dir, orig_files[0]))
        rng = float(a0.max() - a0.min())
    else:
        rng = 0.0
    rel_mae = mae / rng if rng > 0 else 0.0
    return max_abs, mae, rel_mae


def main():
    p = argparse.ArgumentParser(description="Evaluate NPY -> H264 videos -> NPY roundtrip error")
    p.add_argument("input_npy_dir", help="Directory with .npy frames (shape HxWxCxF)")
    p.add_argument("recon_npy_dir", nargs="?", help="Comparison Directory with .npy frames (shape HxWxCxF)")
    args = p.parse_args()

    max_abs, mae, rel_mae = compute_errors(args.input_npy_dir, args.recon_npy_dir)
    print(f"MaxAbsError: {max_abs}")
    print(f"MeanAbsError: {mae}")
    print(f"RelativeMAE: {rel_mae}")

if __name__ == "__main__":
    main()
