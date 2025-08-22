import argparse
import os
import shutil
import tempfile
from typing import Tuple

import numpy as np

def compute_errors(orig_dir: str, recon_dir: str) -> Tuple[float, float, float]:
    orig_files = [f for f in os.listdir(orig_dir) if f.lower().endswith(".npy")]
    recon_files = [f for f in os.listdir(recon_dir) if f.lower().endswith(".npy")]
    orig_files.sort()
    recon_files.sort()
    if orig_files != recon_files:
        raise RuntimeError("Original and reconstructed directories have different file sets")

    print(f"Comparing {len(orig_files)} files in {orig_dir} and {recon_dir}")
    max_abs = 0.0
    mae_sum = 0.0
    count = 0
    for fname in orig_files:
        a = np.load(os.path.join(orig_dir, fname))
        b = np.load(os.path.join(recon_dir, fname))
        print(f"Comparing {fname}:")
        print(f"Shapes: {a.shape} vs {b.shape}, Dtypes: {a.dtype} vs {b.dtype}")
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError(f"Mismatch for {fname}: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")

        # Compute absolute difference only on finite entries to avoid NaN/Inf propagation
        finite_mask = np.isfinite(a) & np.isfinite(b)
        if not np.all(finite_mask):
            bad = a.size - int(finite_mask.sum())
            print(f"Warning: {bad} non-finite elements skipped in {fname}")

        if finite_mask.any():
            diff_vals = np.abs(a[finite_mask] - b[finite_mask])
            file_max = float(diff_vals.max())
            file_mean = float(diff_vals.mean())
        else:
            # No comparable finite values
            file_max = 0.0
            file_mean = 0.0

        max_abs = max(max_abs, file_max)
        mae_sum += file_mean
        print(f"{fname}: MaxAbsError={file_max}, MeanAbsError={file_mean}")
        count += 1
    mae = mae_sum / max(count, 1)
    # Relative MAE normalized by dynamic range (approx. from first file). Avoid NaNs/div by zero.
    if count > 0:
        a0 = np.load(os.path.join(orig_dir, orig_files[0]))
        fm = np.isfinite(a0)
        if fm.any():
            rng = float(np.max(a0[fm]) - np.min(a0[fm]))
        else:
            rng = 0.0
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
