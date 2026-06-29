"""Collect a DRIVING dataset where the actuator gain VARIES PER EPISODE (the hidden-parameter
phase). Each episode draws its own gain from [--gain-lo, --gain-hi]; the true per-step gain is
logged to the dataset's `gains` array and the range to attrs. The model must INFER the gain
fresh each episode from how the motion responds (needs hidden latent dims + frame history),
rather than learning one global constant. Ring marker, same schema as data/collect.py.

Run on the tower (needs the sim, no torch):
    python make_varying_actuator_dataset.py                          # run09, gain ~ U[0.5, 1.0]
    python make_varying_actuator_dataset.py --gain-lo 0.6 --gain-hi 0.9
"""
import sys
import argparse
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as C
from data.collect import collect_dataset


def main():
    p = argparse.ArgumentParser(description="Collect a ring dataset with per-episode varying actuator gain.")
    p.add_argument("--name", default="run09_64x64_varactuator", help="output .h5 stem under datasets/")
    p.add_argument("--grid-size", type=int, default=64)
    p.add_argument("--gain-lo", type=float, default=0.5, help="lower bound of the per-episode gain")
    p.add_argument("--gain-hi", type=float, default=1.0, help="upper bound of the per-episode gain")
    p.add_argument("--n-episodes", type=int, default=C.N_EPISODES)
    p.add_argument("--seed", type=int, default=C.SEED)
    a = p.parse_args()

    out = C.DATASETS_DIR / f"{a.name}.h5"
    print(f"collecting {a.n_episodes} episodes -> {out}  (marker=ring, gain ~ U[{a.gain_lo}, {a.gain_hi}] per episode)")
    collect_dataset(out, n_episodes=a.n_episodes, grid_size=a.grid_size, marker="ring",
                    gain_range=(a.gain_lo, a.gain_hi), seed=a.seed)
    print(f"done -> {out}  (true per-episode gains logged in the 'gains' array)")


if __name__ == "__main__":
    main()
