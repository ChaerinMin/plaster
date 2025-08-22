import json
import math
import os
import sys
import tarfile
import tempfile
import shutil
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import imageio
except Exception as e:  # pragma: no cover - dependency/runtime specific
    raise RuntimeError(
        "imageio is required. Please install with `pip install imageio imageio-ffmpeg`"
    ) from e

def _notify(msg: str) -> None:
    print(msg, flush=True)

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
    # Quantization metadata (present when quantize=True)
    quantize: bool = False
    quant_bits: int = 8
    q_method: str = "linear"
    q_min: List[float] = field(default_factory=list)  # per-channel
    q_max: List[float] = field(default_factory=list)  # per-channel


def _sorted_npy_files(input_dir: str) -> List[str]:
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".npy")]
    files.sort()
    return [os.path.join(input_dir, f) for f in files]


def _open_writer(path: str, fps: int, codec: str, crf: int, pix_fmt: Optional[str], extra_params: Optional[List[str]] = None):
    # Prefer explicit ffmpeg backend
    ffparams = ["-crf", str(crf)]
    if pix_fmt:
        ffparams.extend(["-pix_fmt", pix_fmt])
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
    output_path: str,
    fps: int = 30,
    crf: int = 0,
    codec: str = "libx264",
    pix_fmt: str = "rgb24",
    quantize: Optional[bool] = None,
    quant_bits: int = 8,
) -> str:
    """
    Encode a directory of .npy frames (shape: H x W x C x F, dtype float32/float64)
    into multiple H.264 video files, packing 3 planes per video as RGB.

    If output_path ends with .tar/.tar.gz/.tgz/.tar.bz2/.tar.xz, a single tar archive is created
    containing all the videos and manifest.json. Otherwise, a directory is created at output_path.

    Returns the path to the resulting tar or directory.
    """
    # Decide archive vs directory output
    _lower = output_path.lower()
    _is_tar = _lower.endswith(".tar") or _lower.endswith(".tar.gz") or _lower.endswith(".tgz") \
        or _lower.endswith(".tar.bz2") or _lower.endswith(".tar.xz")

    if _is_tar:
        work_dir = tempfile.mkdtemp(prefix="duv_encode_")
        _notify(f"Output is tar archive. Staging files in {work_dir}")
    else:
        work_dir = output_path
        os.makedirs(work_dir, exist_ok=True)
    npy_paths = _sorted_npy_files(input_dir)
    if not npy_paths:
        raise FileNotFoundError(f"No .npy files found in {input_dir}")

    # Inspect first frame for shape/dtype
    first = np.load(npy_paths[0], mmap_mode=None)
    h, w, c, f, dtype = _validate_and_infer(first)
    if quantize is None:
        quantize = crf > 0  # default: raw bytes for lossless (crf=0), quantize for lossy

    num_bytes = np.dtype(dtype).itemsize

    # Quantization setup (per-channel, across all frames and layers)
    q_min: Optional[np.ndarray] = None
    q_max: Optional[np.ndarray] = None
    if quantize:
        if quant_bits != 8:
            raise ValueError("Only 8-bit quantization is supported currently")
        q_min = np.full((c,), np.inf, dtype=np.float64)
        q_max = np.full((c,), -np.inf, dtype=np.float64)
        _notify(f"Scanning {len(npy_paths)} files to compute per-channel min/max for quantization...")
        _rep_every = max(1, len(npy_paths) // 20)
        for _idx, npy_path in enumerate(npy_paths):
            arr = np.load(npy_path)
            _h, _w, _c, _f, _dt = _validate_and_infer(arr)
            if (_h, _w, _c, _f) != (h, w, c, f) or _dt != dtype:
                raise ValueError(
                    f"Frame {npy_path} has inconsistent shape/dtype: {arr.shape}, {arr.dtype}; expected {(h,w,c,f)}, {dtype}"
                )
            # Update per-channel min/max over all layers and spatial dims
            for ch in range(c):
                vals = arr[:, :, ch, :]
                fm = np.isfinite(vals)
                if fm.any():
                    ch_vals = vals[fm]
                    vmin = float(ch_vals.min())
                    vmax = float(ch_vals.max())
                    if vmin < q_min[ch]:
                        q_min[ch] = vmin
                    if vmax > q_max[ch]:
                        q_max[ch] = vmax
            # periodic progress
            if (_idx % _rep_every) == 0 or _idx == len(npy_paths) - 1:
                _notify(f"  scanned {_idx+1}/{len(npy_paths)} files")
        _notify("Quantization scan complete.")
        # Handle channels that had no finite values
        for ch in range(c):
            if not np.isfinite(q_min[ch]) or not np.isfinite(q_max[ch]):
                q_min[ch] = 0.0
                q_max[ch] = 1.0

    # Planes per layer: either one per byte of each channel, or one per channel when quantized
    planes_per_layer = c if quantize else c * num_bytes
    mapping = _build_group_mapping(planes_per_layer)
    groups_per_layer = len(mapping)

    # Choose codec/pix_fmt combinations and padding rules
    used_codec = codec
    used_pix_fmt: Optional[str] = pix_fmt
    extra_params: List[str] = []
    # Prefer libx264rgb to avoid YUV conversions when using RGB data or quantized path
    if codec == "libx264" and (quantize or (pix_fmt and pix_fmt.startswith("rgb")) or crf == 0):
        used_codec = "libx264rgb"
        # Let encoder choose appropriate output pixel fmt; input is rgb24 via imageio
        used_pix_fmt = None
        # Ensure true lossless and avoid limited range scaling
        extra_params.extend(["-preset", "medium", "-color_range", "pc", "-loglevel", "error"])  # sensible default

    def _needs_even_dims(fmt: Optional[str]) -> bool:
        return bool(fmt) and (fmt.startswith("yuv420") or fmt.startswith("yuv422"))

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
    total_videos = f * groups_per_layer
    _notify(
        f"Encoding {total_videos} videos (layers={f}, groups/layer={groups_per_layer}), frames/video={len(npy_paths)}"
    )
    _vid_idx = 0
    for layer in range(f):
        for g, plane_idxs in enumerate(mapping):
            out_path = os.path.join(work_dir, f"layer{layer:02d}_group{g:02d}.mp4")
            _notify(f"[{_vid_idx+1}/{total_videos}] Encoding layer {layer+1}/{f}, group {g+1}/{groups_per_layer}...")
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

                    if quantize:
                        # Quantize this layer's channels to 8-bit
                        vals = arr[:, :, :, layer]
                        # Handle non-finite by mapping them to 0 after scaling
                        q = np.empty((h, w, c), dtype=np.uint8)
                        for ch in range(c):
                            v = vals[:, :, ch]
                            fm = np.isfinite(v)
                            # Avoid zero range
                            mn = float(q_min[ch])
                            mx = float(q_max[ch])
                            rng = mx - mn
                            if rng <= 0 or not np.isfinite(rng):
                                # Flat channel or invalid range
                                q[:, :, ch] = 0
                            else:
                                scaled = np.zeros_like(v, dtype=np.float32)
                                scaled[fm] = (v[fm] - mn) / rng
                                scaled = np.clip(scaled, 0.0, 1.0)
                                q[:, :, ch] = np.round(scaled * 255.0).astype(np.uint8)
                        planes_stack = q  # H x W x C
                    else:
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
            _notify(f"Completed video {_vid_idx+1}/{total_videos}: {os.path.basename(out_path)}")
            _vid_idx += 1

    manifest = Manifest(
        version="1.0",
        dtype=np.dtype(dtype).str,  # preserve endianness (e.g. '<f4')
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
        quantize=bool(quantize),
        quant_bits=quant_bits if quantize else 0,
        q_method="linear" if quantize else "",
        q_min=(q_min.tolist() if quantize and q_min is not None else []),
        q_max=(q_max.tolist() if quantize and q_max is not None else []),
    )

    manifest_path = os.path.join(work_dir, "manifest.json")
    with open(manifest_path, "w") as fobj:
        json.dump(asdict(manifest), fobj, indent=2)

    # If tar requested, pack files and cleanup staging directory
    if _is_tar:
        # Choose compression mode
        if _lower.endswith(".tar.gz") or _lower.endswith(".tgz"):
            mode = "w:gz"
        elif _lower.endswith(".tar.bz2"):
            mode = "w:bz2"
        elif _lower.endswith(".tar.xz"):
            mode = "w:xz"
        else:
            mode = "w"
        tar_path = output_path
        _notify(f"Creating archive: {tar_path}")
        with tarfile.open(tar_path, mode) as tf:
            for name in sorted(os.listdir(work_dir)):
                full = os.path.join(work_dir, name)
                tf.add(full, arcname=name)
        shutil.rmtree(work_dir, ignore_errors=True)
        _notify(f"Encoding complete. Archive written to: {tar_path}")
        return tar_path
    else:
        _notify(f"Encoding complete. Manifest written to: {manifest_path}")
        return manifest_path


def decode_videos_to_npy_dir(video_source: str, output_dir: str) -> None:
    """
    Decode videos produced by encode_npy_dir_to_videos back to a directory of .npy files.
    video_source may be either a directory containing manifest.json and videos, or
    a tar archive (.tar/.tar.gz/.tgz/.tar.bz2/.tar.xz) containing those files.
    """
    # Determine working directory (extract tar if needed)
    cleanup_dir: Optional[str] = None
    if os.path.isdir(video_source):
        work_dir = video_source
    elif os.path.isfile(video_source):
        lower = video_source.lower()
        is_tar = lower.endswith(".tar") or lower.endswith(".tar.gz") or lower.endswith(".tgz") \
            or lower.endswith(".tar.bz2") or lower.endswith(".tar.xz")
        if not is_tar:
            raise FileNotFoundError(f"Unsupported input file (expecting tar archive): {video_source}")
        work_dir = tempfile.mkdtemp(prefix="duv_decode_")
        cleanup_dir = work_dir
        _notify(f"Extracting archive to {work_dir}")
        # Detect compression automatically by mode 'r:*'
        with tarfile.open(video_source, "r:*") as tf:
            tf.extractall(work_dir)
    else:
        raise FileNotFoundError(f"Input path not found: {video_source}")

    manifest_path = os.path.join(work_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest.json not found in {work_dir}")
    with open(manifest_path, "r") as fobj:
        man = Manifest(**json.load(fobj))

    os.makedirs(output_dir, exist_ok=True)

    out_h = man.height + man.pad_h
    out_w = man.width + man.pad_w
    dtype = np.dtype(man.dtype)
    num_bytes = man.itemsize

    _notify(
        f"Decoding {man.num_frames} frames (layers={man.layers}, groups/layer={man.groups_per_layer}) from {work_dir}"
    )
    _report_every = max(1, man.num_frames // 20)
    for t in range(man.num_frames):
        if (t % _report_every) == 0 or t == man.num_frames - 1:
            _notify(f"Decoding frame {t+1}/{man.num_frames}")
        # If quantized, collect uint8 per-channel; else, collect raw bytes
        if man.quantize:
            frame_q = np.empty((man.height, man.width, man.channels, man.layers), dtype=np.uint8)
        else:
            # Buffer for this frame's bytes: (H, W, C, F, num_bytes)
            frame_bytes = np.empty((man.height, man.width, man.channels, man.layers, num_bytes), dtype=np.uint8)

        for layer in range(man.layers):
            # Collect all planes for this layer: (H, W, planes_per_layer)
            planes_stack = np.empty((man.height, man.width, man.planes_per_layer), dtype=np.uint8)

            for g, plane_idxs in enumerate(man.group_mapping):
                path = os.path.join(work_dir, f"layer{layer:02d}_group{g:02d}.mp4")
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

            if man.quantize:
                # planes_stack is (H, W, C) in quantized mode
                frame_q[:, :, :, layer] = planes_stack
            else:
                # Reshape back to (H, W, C, num_bytes) for this layer and place into frame_bytes
                layer_bytes = planes_stack.reshape(man.height, man.width, man.channels, num_bytes)
                frame_bytes[:, :, :, layer, :] = layer_bytes

        if man.quantize:
            # Dequantize back to float array
            qmin = np.asarray(man.q_min, dtype=np.float64)
            qmax = np.asarray(man.q_max, dtype=np.float64)
            rng = qmax - qmin
            rng[rng <= 0] = 1.0
            # Scale from [0,255] to [qmin, qmax]
            arr = frame_q.astype(np.float32) / 255.0
            for ch in range(man.channels):
                arr[:, :, ch, :] = arr[:, :, ch, :] * rng[ch] + qmin[ch]
            # Cast to original dtype if desired (keep float32 for stability)
            if dtype == np.float64:
                arr = arr.astype(np.float64)
        else:
            # View back to original dtype and shape
            arr = frame_bytes.view(dtype).reshape(man.height, man.width, man.channels, man.layers)
        out_name = man.frame_files[t]
        out_path = os.path.join(output_dir, out_name)
        np.save(out_path, arr)

    # Readers are opened and closed per access above.
    _notify(f"Decoding complete. NPY files written to: {output_dir}")
    if cleanup_dir:
        shutil.rmtree(cleanup_dir, ignore_errors=True)

