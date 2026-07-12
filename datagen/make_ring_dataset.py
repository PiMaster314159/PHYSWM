"""Collect a DRIVING dataset with the new 'ring' marker (white body, grey heading ring
around the front tip). Same random-driving policy and HDF5 schema as data/collect.py, so
make_dataloaders / the probes / the runners read it unchanged.

Run on the tower (needs the sim, no torch):
    python make_ring_dataset.py                 # defaults: run07_64x64_ring, 64x64
    python make_ring_dataset.py --n-episodes 8000 --grid-size 64
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
    p = argparse.ArgumentParser(description="Collect a driving dataset with the ring marker.")
    p.add_argument("--name", default="run07_64x64_ring", help="output .h5 stem under datasets/")
    p.add_argument("--grid-size", type=int, default=64)
    p.add_argument("--n-episodes", type=int, default=C.N_EPISODES)
    p.add_argument("--seed", type=int, default=C.SEED)
    a = p.parse_args()

    out = C.DATASETS_DIR / f"{a.name}.h5"
    print(f"collecting {a.n_episodes} driving episodes -> {out}  (marker=ring, grid={a.grid_size})")
    collect_dataset(out, n_episodes=a.n_episodes, grid_size=a.grid_size, marker="ring", seed=a.seed)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
