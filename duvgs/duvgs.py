import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import imageio
except Exception as e:  # pragma: no cover - dependency/runtime specific
    raise RuntimeError(
        "imageio is required. Please install with `pip install imageio imageio-ffmpeg`"
    ) from e

@dataclass
class Manifest:
    version: str
    dtype: str
    itemsize: int
    byteorder: str
    height: int
    width: int
    channels: int
    layers: int
    num_frames: int
    fps: int
    codec: str
    crf: int
    pix_fmt: str
    pad_h: int
    pad_w: int
    groups_per_layer: int
    planes_per_layer: int
    group_mapping: List[List[int]]  # per group, list of plane indices (<=3, -1 means padded)
    frame_files: List[str]


def _sorted_npy_files(input_dir: str) -> List[str]:
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".npy")]
    files.sort()
    return [os.path.join(input_dir, f) for f in files]


def _open_writer(path: str, fps: int, codec: str, crf: int, pix_fmt: str, extra_params: Optional[List[str]] = None):
    # Prefer explicit ffmpeg backend
    ffparams = ["-crf", str(crf), "-pix_fmt", pix_fmt]
    if extra_params:
        ffparams.extend(extra_params)
    params = dict(fps=fps, codec=codec, format="ffmpeg", ffmpeg_params=ffparams)
    try:
        return imageio.get_writer(path, **params)
    except TypeError:
        # Older imageio may not support format kwarg
        params.pop("format", None)
        return imageio.get_writer(path, **params)


def _open_reader(path: str):
    try:
        return imageio.get_reader(path, format="ffmpeg")
    except TypeError:
        return imageio.get_reader(path)


def _build_group_mapping(planes_per_layer: int) -> List[List[int]]:
    groups = math.ceil(planes_per_layer / 3)
    mapping: List[List[int]] = []
    for g in range(groups):
        start = g * 3
        group = []
        for k in range(3):
            idx = start + k
            group.append(idx if idx < planes_per_layer else -1)
        mapping.append(group)
    return mapping


def _validate_and_infer(arr: np.ndarray) -> Tuple[int, int, int, int, np.dtype]:
    if arr.ndim != 4:
        raise ValueError(f"Expected array of shape (H, W, C, F), got {arr.shape}")
    h, w, c, f = arr.shape
    if c <= 0 or f <= 0:
        raise ValueError("Channels and layers must be > 0")
    if arr.dtype not in (np.float32, np.float64):
        raise ValueError(f"Expected dtype float32 or float64, got {arr.dtype}")
    return h, w, c, f, arr.dtype


def encode_npy_dir_to_videos(
    input_dir: str,
    output_dir: str,
    fps: int = 30,
    crf: int = 18,
    codec: str = "libx264",
    pix_fmt: str = "yuv420p",
) -> str:
    """
    Encode a directory of .npy frames (shape: H x W x C x F, dtype float32/float64)
    into multiple H.264 video files, packing 3 byte-planes per video as RGB.

    Returns the path to the manifest.json written in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    npy_paths = _sorted_npy_files(input_dir)
    if not npy_paths:
        raise FileNotFoundError(f"No .npy files found in {input_dir}")

    # Inspect first frame for shape/dtype
    first = np.load(npy_paths[0], mmap_mode=None)
    h, w, c, f, dtype = _validate_and_infer(first)
    num_bytes = np.dtype(dtype).itemsize
    planes_per_layer = c * num_bytes
    mapping = _build_group_mapping(planes_per_layer)
    groups_per_layer = len(mapping)

    # Choose codec/pix_fmt combinations and padding rules
    used_codec = codec
    used_pix_fmt = pix_fmt
    extra_params: List[str] = []
    # Use libx264rgb for lossless (crf=0) to avoid subsampling and ensure compatibility
    if crf == 0 and codec == "libx264":
        used_codec = "libx264rgb"
        used_pix_fmt = "rgb24"
        extra_params.extend(["-preset", "medium"])  # sensible default

    def _needs_even_dims(fmt: str) -> bool:
        return fmt.startswith("yuv420") or fmt.startswith("yuv422")

    if _needs_even_dims(used_pix_fmt):
        pad_h = h % 2
        pad_w = w % 2
    else:
        pad_h = 0
        pad_w = 0
    out_h = h + pad_h
    out_w = w + pad_w

    # Stream frames, but avoid opening all writers at once to prevent encoder errors.
    # Process one (layer, group) video at a time.
    for layer in range(f):
        for g, plane_idxs in enumerate(mapping):
            out_path = os.path.join(output_dir, f"layer{layer:02d}_group{g:02d}.mp4")
            writer = _open_writer(
                out_path, fps=fps, codec=used_codec, crf=crf, pix_fmt=used_pix_fmt, extra_params=extra_params
            )
            try:
                for frame_idx, npy_path in enumerate(npy_paths):
                    arr = np.load(npy_path)
                    _h, _w, _c, _f, _dt = _validate_and_infer(arr)
                    if (_h, _w, _c, _f) != (h, w, c, f) or _dt != dtype:
                        raise ValueError(
                            f"Frame {npy_path} has inconsistent shape/dtype: {arr.shape}, {arr.dtype}; expected {(h,w,c,f)}, {dtype}"
                        )

                    arr = np.ascontiguousarray(arr)
                    byte_view = arr.view(np.uint8).reshape(h, w, c, f, num_bytes)
                    planes_stack = byte_view[:, :, :, layer, :].reshape(h, w, planes_per_layer)

                    # Assemble RGB frame
                    if pad_h or pad_w:
                        frame_rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                        target = frame_rgb[:h, :w]
                    else:
                        frame_rgb = None
                        target = None

                    channels = []
                    for k in range(3):
                        pidx = plane_idxs[k]
                        if pidx >= 0:
                            ch = planes_stack[:, :, pidx]
                        else:
                            ch = np.zeros((h, w), dtype=np.uint8)
                        channels.append(ch)

                    rgb = np.stack(channels, axis=2)
                    if target is not None:
                        target[:, :, :] = rgb
                        to_write = frame_rgb
                    else:
                        to_write = rgb

                    writer.append_data(to_write)
            finally:
                try:
                    writer.close()
                except Exception:
                    pass

    manifest = Manifest(
        version="1.0",
        dtype=str(np.dtype(dtype)),
        itemsize=num_bytes,
        byteorder=sys.byteorder,
        height=h,
        width=w,
        channels=c,
        layers=f,
        num_frames=len(npy_paths),
    fps=fps,
    codec=used_codec,
    crf=crf,
    pix_fmt=used_pix_fmt,
        pad_h=pad_h,
        pad_w=pad_w,
        groups_per_layer=groups_per_layer,
        planes_per_layer=planes_per_layer,
        group_mapping=mapping,
        frame_files=[os.path.basename(p) for p in npy_paths],
    )

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as fobj:
        json.dump(asdict(manifest), fobj, indent=2)

    return manifest_path


def decode_videos_to_npy_dir(video_dir: str, output_dir: str) -> None:
    """
    Decode videos produced by encode_npy_dir_to_videos back to a directory of .npy files.
    """
    manifest_path = os.path.join(video_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest.json not found in {video_dir}")
    with open(manifest_path, "r") as fobj:
        man = Manifest(**json.load(fobj))

    os.makedirs(output_dir, exist_ok=True)

    out_h = man.height + man.pad_h
    out_w = man.width + man.pad_w
    dtype = np.dtype(man.dtype)
    num_bytes = man.itemsize

    for t in range(man.num_frames):
        # Buffer for this frame's bytes: (H, W, C, F, num_bytes)
        frame_bytes = np.empty((man.height, man.width, man.channels, man.layers, num_bytes), dtype=np.uint8)

        for layer in range(man.layers):
            # Collect all planes for this layer: (H, W, planes_per_layer)
            planes_stack = np.empty((man.height, man.width, man.planes_per_layer), dtype=np.uint8)

            for g, plane_idxs in enumerate(man.group_mapping):
                path = os.path.join(video_dir, f"layer{layer:02d}_group{g:02d}.mp4")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing video file: {path}")
                rdr = _open_reader(path)
                try:
                    frame_rgb = rdr.get_data(t)
                finally:
                    try:
                        rdr.close()
                    except Exception:
                        pass
                if frame_rgb.shape[0] != out_h or frame_rgb.shape[1] != out_w:
                    raise ValueError(
                        f"Unexpected frame size in layer {layer} group {g}: {frame_rgb.shape}; expected {(out_h, out_w)}"
                    )
                rgb = frame_rgb[: man.height, : man.width, :]
                for k in range(3):
                    pidx = plane_idxs[k]
                    if pidx >= 0:
                        planes_stack[:, :, pidx] = rgb[:, :, k]

            # Reshape back to (H, W, C, num_bytes) for this layer and place into frame_bytes
            layer_bytes = planes_stack.reshape(man.height, man.width, man.channels, num_bytes)
            frame_bytes[:, :, :, layer, :] = layer_bytes

        # View back to original dtype and shape
        arr = frame_bytes.view(dtype).reshape(man.height, man.width, man.channels, man.layers)
        out_name = man.frame_files[t]
        out_path = os.path.join(output_dir, out_name)
        np.save(out_path, arr)

    # Readers are opened and closed per access above.


# Write a basic usage example. Use argparse
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode NPY files to H.264 videos")
    parser.add_argument("input_dir", help="Input directory containing .npy files")
    parser.add_argument("output_dir", help="Output directory for H.264 videos")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--crf", type=int, default=23, help="Constant Rate Factor")
    parser.add_argument("--codec", default="libx264", help="Video codec")
    parser.add_argument("--pix_fmt", default="yuv420p", help="Pixel format")
    args = parser.parse_args()

    encode_npy_dir_to_videos(
        args.input_dir,
        args.output_dir,
        fps=args.fps,
        crf=args.crf,
        codec=args.codec,
        pix_fmt=args.pix_fmt,
    )
