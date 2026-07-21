"""Straight-track MPC demo: a unicycle robot starts at the WRONG HEADING and MPPI steers it
onto the centerline and down the track, staying inside the width.

Step 1 of the MPC build order (see memory/MPC_cheatsheet.md): the world model here is the
gray-box unicycle dynamics itself (known kinematics), planned with MPPI. Later we swap this
`step` for the neural world model and close the loop from pixels; the controller is unchanged.

State s = [x, y, theta].  Action a = [omega] (steering rate); forward speed v is fixed.
Track: centerline y = 0, drive along +x, stay within |y| <= W/2.

    python run_track_mpc.py                       # default 45-degree start
    python run_track_mpc.py --theta0-deg 70 --y0 0.2
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import argparse
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from control.mppi import mppi_plan


# ---- gray-box world model: semi-implicit unicycle, vectorized over samples ----
def make_step(dt, v):
    def step(S, A):                       # S (K,3)=[x,y,theta], A (K,1)=[omega]
        x, y, th = S[:, 0], S[:, 1], S[:, 2]
        omega = A[:, 0]
        th_new = th + omega * dt          # rotate heading first (semi-implicit)
        x_new = x + v * np.cos(th_new) * dt
        y_new = y + v * np.sin(th_new) * dt
        return np.stack([x_new, y_new, th_new], axis=1)
    return step


def wrap(a):                              # wrap angle to [-pi, pi]
    return np.arctan2(np.sin(a), np.cos(a))


# ---- cost: stay on centerline, face down-track, gentle steering, soft track bound ----
def make_costs(W, w_lat, w_head, w_ctrl, w_bound, term_scale):
    half = W / 2.0
    def running(S, A, S_next=None):
        y, th, omega = S[:, 1], S[:, 2], A[:, 0]
        viol = np.maximum(0.0, np.abs(y) - half)
        return (w_lat * y**2 + w_head * wrap(th)**2 + w_ctrl * omega**2
                + w_bound * viol**2)
    def terminal(S):
        y, th = S[:, 1], S[:, 2]
        viol = np.maximum(0.0, np.abs(y) - half)
        return term_scale * (w_lat * y**2 + w_head * wrap(th)**2) + w_bound * viol**2
    return running, terminal


def parse_args():
    p = argparse.ArgumentParser(description="Straight-track MPPI control demo.")
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--v", type=float, default=1.0, help="fixed forward speed")
    p.add_argument("--track-width", type=float, default=1.0)
    p.add_argument("--x0", type=float, default=0.0)
    p.add_argument("--y0", type=float, default=0.0)
    p.add_argument("--theta0-deg", type=float, default=45.0, help="initial (wrong) heading")
    p.add_argument("--horizon", type=int, default=25, help="MPPI planning horizon H")
    p.add_argument("--samples", type=int, default=1000, help="MPPI rollouts K")
    p.add_argument("--sigma", type=float, default=0.8, help="action exploration std (rad/s)")
    p.add_argument("--lam", type=float, default=1.0, help="MPPI temperature")
    p.add_argument("--omega-max", type=float, default=2.5, help="steering-rate bound")
    p.add_argument("--steps", type=int, default=70, help="closed-loop control steps")
    p.add_argument("--w-lat", type=float, default=6.0)
    p.add_argument("--w-head", type=float, default=1.0)
    p.add_argument("--w-ctrl", type=float, default=0.05)
    p.add_argument("--w-bound", type=float, default=60.0)
    p.add_argument("--term-scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    rng = np.random.default_rng(a.seed)
    step = make_step(a.dt, a.v)
    running, terminal = make_costs(a.track_width, a.w_lat, a.w_head, a.w_ctrl, a.w_bound, a.term_scale)
    a_low = np.array([-a.omega_max]); a_high = np.array([a.omega_max])

    s = np.array([a.x0, a.y0, np.deg2rad(a.theta0_deg)], float)
    a_nom = np.zeros((a.horizon, 1))
    traj = [s.copy()]
    applied = []
    for _ in range(a.steps):
        a_nom, info = mppi_plan(s, a_nom, step, running, terminal,
                                n_samples=a.samples, sigma=a.sigma, lam=a.lam,
                                a_low=a_low, a_high=a_high, rng=rng)
        a0 = a_nom[0].copy()
        s = step(s[None], a0[None])[0]            # apply first action to the TRUE system
        traj.append(s.copy()); applied.append(a0[0])
        a_nom = np.roll(a_nom, -1, axis=0); a_nom[-1] = 0.0   # warm-start shift

    traj = np.array(traj)                          # (T+1, 3)
    half = a.track_width / 2.0
    x, y, th = traj[:, 0], traj[:, 1], traj[:, 2]
    final_lat = abs(y[-1]); final_head = abs(np.rad2deg(wrap(th[-1])))
    max_excursion = np.abs(y).max()
    print(f"start: y={a.y0:.2f}, theta={a.theta0_deg:.0f} deg")
    print(f"final: |lateral|={final_lat:.3f}, |heading|={final_head:.1f} deg")
    print(f"max |y| along run = {max_excursion:.3f}  (track half-width {half:.2f}) "
          f"-> {'STAYED IN BOUNDS' if max_excursion <= half + 1e-6 else 'LEFT TRACK'}")

    # ---- figure: path with track, plus lateral + heading vs time ----
    report = ROOT / "results" / "track_mpc"; report.mkdir(parents=True, exist_ok=True)
    fig, (axp, axt) = plt.subplots(1, 2, figsize=(12, 4.2))

    axp.axhspan(-half, half, color="0.9", label="track")
    axp.axhline(0, color="0.6", ls="--", lw=1)
    axp.axhline(half, color="0.4", lw=1); axp.axhline(-half, color="0.4", lw=1)
    axp.plot(x, y, "-o", ms=3, color="#2c7fb8", label="robot path")
    L = 0.35
    axp.annotate("", xy=(x[0] + L*np.cos(th[0]), y[0] + L*np.sin(th[0])), xytext=(x[0], y[0]),
                 arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2))
    axp.plot(x[0], y[0], "s", color="#c0392b", ms=7, label="start (wrong heading)")
    axp.set_xlabel("x (down-track)"); axp.set_ylabel("y (lateral)")
    axp.set_title("MPPI steers onto the track"); axp.legend(fontsize=8, loc="upper right")
    axp.set_ylim(-max(half*1.6, max_excursion*1.2), max(half*1.6, max_excursion*1.2))

    t = np.arange(len(y)) * a.dt
    axt.axhline(0, color="0.6", ls="--", lw=1)
    axt.plot(t, y, color="#2c7fb8", label="lateral offset y")
    axt.plot(t, np.rad2deg(wrap(th)) / 90.0, color="#e08214", label="heading err (/90 deg)")
    axt.set_xlabel("time (s)"); axt.set_title("error vs time (both -> 0)")
    axt.legend(fontsize=8)

    fig.tight_layout()
    out = report / f"track_mpc_theta{int(a.theta0_deg)}_y{a.y0:g}.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
