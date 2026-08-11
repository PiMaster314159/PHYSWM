"""Re-encode an existing dataset's `frames` from float32 to uint8, in place of nothing -- the source
file is never modified; a new .h5 is written beside it.

Why: the renderer emits [0,1] with very few distinct levels (a ring marker is exactly 0 / 0.5 / 1), so
float32 spent 32 bits carrying under 2 bits of signal. The 1.3M-frame 64x64 bicycle set is 19.5 GB as
float32 and 4.9 GB as uint8 -- the difference between "does not fit in host RAM" and "loads once and
trains with no per-sample decompression".

Frames are stored as round(value * frame_scale) with frame_scale=255, and the scale is recorded as a
file attribute so readers divide it back out. Datasets written before this convention have no such
attribute and are read at scale 1.0, so old files keep working untouched.

Conversion streams in blocks and never holds the whole array, so it runs in a few hundred MB
regardless of dataset size.

    python datagen/convert_frames_uint8.py --in data/datasets/64x64_bicycle.h5
    python datagen/convert_frames_uint8.py --in ... --out ... --replace
"""
import sys
import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FRAME_SCALE


def convert(src: Path, dst: Path, block: int = 8192) -> None:
    with h5py.File(src, "r") as f_in, h5py.File(dst, "w") as f_out:
        frames_in = f_in["frames"]
        n = frames_in.shape[0]
        scale_in = float(f_in.attrs.get("frame_scale", 1.0))
        if scale_in != 1.0:
            raise SystemExit(f"{src.name} already has frame_scale={scale_in}; nothing to do")

        lo, hi = float(frames_in[:block].min()), float(frames_in[:block].max())
        if hi > 1.0 + 1e-6 or lo < -1e-6:
            raise SystemExit(f"expected frames in [0,1], found [{lo}, {hi}] -- refusing to quantize")

        frames_out = f_out.create_dataset(
            "frames", shape=frames_in.shape, dtype="uint8",
            chunks=frames_in.chunks, compression=frames_in.compression,
        )
        for i in range(0, n, block):
            chunk = np.asarray(frames_in[i:i + block], dtype=np.float32)
            frames_out[i:i + block] = np.rint(chunk * FRAME_SCALE).astype(np.uint8)
            done = min(i + block, n)
            print(f"\r  frames {done}/{n}  ({100.0 * done / n:5.1f}%)", end="", flush=True)
        print()

        for k in f_in.keys():                      # everything else is small: copy verbatim
            if k != "frames":
                f_in.copy(k, f_out)
        for k, v in f_in.attrs.items():
            f_out.attrs[k] = v
        f_out.attrs["frame_scale"] = FRAME_SCALE


def main():
    p = argparse.ArgumentParser(description="Re-encode dataset frames as uint8 (4x smaller, same pixels).")
    p.add_argument("--in", dest="src", required=True, help="source .h5")
    p.add_argument("--out", dest="dst", default=None, help="destination .h5 (default: <src>_u8.h5)")
    p.add_argument("--block", type=int, default=8192, help="frames converted per read")
    p.add_argument("--replace", action="store_true",
                   help="after a successful convert, move the new file over the original")
    a = p.parse_args()

    src = Path(a.src)
    dst = Path(a.dst) if a.dst else src.with_name(src.stem + "_u8.h5")
    if not src.exists():
        raise SystemExit(f"no such file: {src}")
    if dst.exists():
        raise SystemExit(f"refusing to overwrite existing {dst}")

    before = src.stat().st_size
    print(f"converting {src.name} -> {dst.name}")
    convert(src, dst, block=a.block)
    after = dst.stat().st_size
    print(f"  {before / 1e9:.2f} GB -> {after / 1e9:.2f} GB  ({before / max(after, 1):.2f}x smaller)")

    if a.replace:
        shutil.move(str(dst), str(src))
        print(f"  replaced {src.name}")


if __name__ == "__main__":
    main()
