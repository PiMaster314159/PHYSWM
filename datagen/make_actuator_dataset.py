"""Collect a DRIVING dataset with an UNMODELED actuator resistance: the stored action is
the commanded speed v, but the sim applies actuator_gain * v (a fixed, unknown efficiency
< 1). The gray-box model is given the command and must recover the gain as its learnable
a_v coefficient. Uses the ring marker, same schema as data/collect.py.

Run on the tower (needs the sim, no torch):
    python make_actuator_dataset.py                       # run08_64x64_actuator, gain 0.7
    python make_actuator_dataset.py --gain 0.6 --name run08_64x64_actuator06
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
    p = argparse.ArgumentParser(description="Collect a ring dataset with actuator resistance.")
    p.add_argument("--name", default="run08_64x64_actuator", help="output .h5 stem under datasets/")
    p.add_argument("--grid-size", type=int, default=64)
    p.add_argument("--gain", type=float, default=0.7, help="actuator efficiency: applied speed = gain * commanded")
    p.add_argument("--n-episodes", type=int, default=C.N_EPISODES)
    p.add_argument("--seed", type=int, default=C.SEED)
    a = p.parse_args()

    out = C.DATASETS_DIR / f"{a.name}.h5"
    print(f"collecting {a.n_episodes} episodes -> {out}  (marker=ring, grid={a.grid_size}, actuator_gain={a.gain})")
    collect_dataset(out, n_episodes=a.n_episodes, grid_size=a.grid_size,
                    marker="ring", actuator_gain=a.gain, seed=a.seed)
    print(f"done -> {out}  (a_v should recover ~{a.gain})")


if __name__ == "__main__":
    main()
