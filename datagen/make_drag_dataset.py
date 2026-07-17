"""Collect a DRIVING dataset with UNMODELED aerodynamic DRAG: a v^2 speed loss.

The sim applies (v - drag_c * v^2) but stores the *commanded* v, so the model sees the command
and must recover the drag. Unlike a constant actuator gain (a single a_v), drag is NONLINEAR in
v, so only a v^2 residual term can capture it — this is the test the structured residual is for.

The speed range is raised on purpose (v_mean 0.30): (a) so the training distribution COVERS the
MPC control speed (v=0.4) instead of sitting at ~0.18, and (b) so the v^2 drag is actually
exercised — at low speed c*v^2 is tiny and there's nothing to learn.

    python datagen/make_drag_dataset.py                          # 64x64_drag, drag_c 1.0
    python datagen/make_drag_dataset.py --drag-c 0.6 --name 64x64_drag06
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
    p = argparse.ArgumentParser(description="Collect a ring driving dataset with v^2 aerodynamic drag.")
    p.add_argument("--name", default="64x64_drag", help="output .h5 stem under datasets/")
    p.add_argument("--grid-size", type=int, default=64)
    p.add_argument("--drag-c", type=float, default=1.0, help="drag coefficient: applied speed = v - drag_c*v^2")
    p.add_argument("--v-mean", type=float, default=0.30, help="raised so training covers the MPC speed (~0.4) and exercises the v^2 drag")
    p.add_argument("--v-std", type=float, default=0.12, help="widened so the speed range spans roughly [0.05, 0.55]")
    p.add_argument("--n-episodes", type=int, default=C.N_EPISODES)
    p.add_argument("--seed", type=int, default=C.SEED)
    a = p.parse_args()

    out = C.DATASETS_DIR / f"{a.name}.h5"
    print(f"collecting {a.n_episodes} episodes -> {out}  (marker=ring, grid={a.grid_size}, "
          f"drag_c={a.drag_c}, v_mean={a.v_mean}, v_std={a.v_std})")
    collect_dataset(out, n_episodes=a.n_episodes, grid_size=a.grid_size, marker="ring",
                    drag_c=a.drag_c, v_mean=a.v_mean, v_std=a.v_std, seed=a.seed)


if __name__ == "__main__":
    main()
