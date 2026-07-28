"""How visually different is each frame from the last? For history/velocity to be readable, consecutive
frames must actually MOVE, and the amount of motion should track the (hidden) speed. This measures the
frame-to-frame pixel change within episodes, correlates it with the true velocity (bicycle datasets), and
saves example consecutive-frame strips so you can eyeball it.

    python analysis/inspect_frames.py --run 64x64_bicycle
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import argparse
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import config as C


def main():
    ap = argparse.ArgumentParser(description="Frame-to-frame visual difference + velocity correlation.")
    ap.add_argument("--run", default="64x64_bicycle")
    ap.add_argument("--max-pairs", type=int, default=200000, help="cap consecutive-frame pairs sampled")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    path = C.DATASETS_DIR / f"{a.run}.h5"
    out = Path(a.out) if a.out else ROOT / "results" / f"frames_{a.run}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "r") as f:
        frames = f["frames"][:]                              # (T, H, W)
        starts = f["episode_starts"][:]; lengths = f["episode_lengths"][:]
        vel = f["velocities"][:] if "velocities" in f else None
        grid = int(f.attrs["grid_size"]); dt = float(f.attrs["dt"]); hold_k = int(f.attrs["hold_k"])
    T = frames.shape[0]
    fr = frames.astype(np.float32)
    if fr.max() > 1.5:
        fr /= 255.0                                          # binary frames stored as uint8

    # consecutive (t, t+1) pairs that stay INSIDE an episode
    idx = []
    for s, n in zip(starts, lengths):
        idx.append(np.arange(int(s), int(s) + int(n) - 1))
    idx = np.concatenate(idx)
    if len(idx) > a.max_pairs:
        idx = np.random.default_rng(0).choice(idx, a.max_pairs, replace=False)

    diff = np.abs(fr[idx + 1] - fr[idx])                     # (P, H, W)
    mean_abs = diff.reshape(len(idx), -1).mean(1)            # mean pixel change per pair
    changed = (diff > 0.05).reshape(len(idx), -1).mean(1) * grid * grid   # # pixels that changed
    on = (fr[idx] > 0.5).reshape(len(idx), -1).sum(1).mean()  # avg lit pixels (object size)

    print(f"=== frame differences: {a.run}  ({T} transitions, grid {grid}, dt {dt}, hold_k {hold_k}) ===")
    print(f"  object size (lit px)      ~ {on:.0f}")
    print(f"  changed px per step       median {np.median(changed):.1f}   mean {changed.mean():.1f}   "
          f"(as % of object: {100*changed.mean()/max(on,1):.0f}%)")
    print(f"  mean |Δpixel| per step    median {np.median(mean_abs):.4f}   mean {mean_abs.mean():.4f}")
    frac_static = (changed < 1.0).mean()
    print(f"  near-static pairs (<1 px change): {100*frac_static:.1f}%")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].hist(changed, bins=60, color="#2a78d6"); axes[0].set_title("changed pixels / step")
    axes[0].set_xlabel("# pixels that changed"); axes[0].set_ylabel("count")

    if vel is not None:
        v_pair = vel[idx]                                    # true speed at t
        axes[1].scatter(v_pair, changed, s=3, alpha=0.15, color="#1baf7a")
        cc = np.corrcoef(v_pair, changed)[0, 1]
        axes[1].set_title(f"motion vs true speed  (corr {cc:+.2f})")
        axes[1].set_xlabel("true velocity v"); axes[1].set_ylabel("changed pixels / step")
        print(f"  corr(changed px, true v)  = {cc:+.3f}   <- if high, speed IS readable from frame motion")
        # expected pixel travel: v*dt*grid
        print(f"  expected travel at v=0.25 : {0.25*dt*grid:.1f} px/step  (object moves this far between frames)")
    else:
        axes[1].axis("off"); axes[1].set_title("(no velocities: unicycle dataset)")

    # example consecutive strip: pick a fast pair so motion is visible
    order = np.argsort(changed)[::-1]
    k = idx[order[len(order) // 50]]                         # a fairly-fast pair
    strip = np.concatenate([fr[k], fr[k + 1], np.abs(fr[k + 1] - fr[k])], axis=1)
    axes[2].imshow(strip, cmap="gray"); axes[2].set_title("frame t | t+1 | |diff|"); axes[2].axis("off")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
