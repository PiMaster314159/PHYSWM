"""Collect a KINEMATIC BICYCLE + THROTTLE dataset. This is the first task where VELOCITY IS A HIDDEN
STATE: the action is (a, delta) = (acceleration, steering angle), velocity integrates the throttle
(v_dot = a), and steering scales with speed (theta_dot = v/L * tan(delta)). Because velocity never
appears in a single rendered frame, the model must infer it from the last few frames -- which is what
finally makes the --n-frames history plumbing load-bearing.

We start with NO drag (drag_c = 0, the prof's clean v_dot = a) to isolate one question: can the model
read hidden velocity off a frame stack and control it? Drag on the velocity dynamics comes later.

The true (hidden) velocity is stored in a `velocities` array for eval only -- never rendered, never fed
to the model.

    python datagen/make_bicycle_dataset.py                       # 64x64_bicycle
    python datagen/make_bicycle_dataset.py --v-max 0.5 --name 64x64_bicycle_slow
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
    p = argparse.ArgumentParser(description="Collect a kinematic-bicycle + throttle dataset (velocity is hidden).")
    p.add_argument("--name", default="64x64_bicycle", help="output .h5 stem under datasets/")
    p.add_argument("--grid-size", type=int, default=64)
    p.add_argument("--v-mean", type=float, default=0.25, help="target cruise speed the throttle controller aims for")
    p.add_argument("--v-std", type=float, default=0.10, help="spread of target speeds (velocity coverage)")
    p.add_argument("--v-max", type=float, default=0.6, help="speed clamp")
    p.add_argument("--a-max", type=float, default=1.5, help="acceleration (throttle) magnitude limit")
    p.add_argument("--delta-max", type=float, default=0.6, help="steering-angle limit (rad)")
    p.add_argument("--wheelbase", type=float, default=C.WHEELBASE)
    p.add_argument("--n-episodes", type=int, default=C.N_EPISODES)
    p.add_argument("--seed", type=int, default=C.SEED)
    a = p.parse_args()

    out = C.DATASETS_DIR / f"{a.name}.h5"
    print(f"collecting {a.n_episodes} bicycle episodes -> {out}  (marker=ring, grid={a.grid_size}, "
          f"v_mean={a.v_mean}, v_std={a.v_std}, v_max={a.v_max}, wheelbase={a.wheelbase}, NO drag)")
    collect_dataset(out, n_episodes=a.n_episodes, grid_size=a.grid_size, marker="ring",
                    dynamics="bicycle", v_mean=a.v_mean, v_std=a.v_std, v_max=a.v_max,
                    a_max=a.a_max, delta_max=a.delta_max, wheelbase=a.wheelbase, seed=a.seed)


if __name__ == "__main__":
    main()
