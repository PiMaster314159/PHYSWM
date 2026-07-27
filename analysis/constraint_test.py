"""Near-edge hard-constraint test: does giving the controller SPEED as a control DOF keep it inside
the track where fixed speed can't? (Turning radius = v/omega_max, so slowing tightens the turn.)

Reads two y0-sweeps produced under --hard-bound -- one fixed-speed, one --plan-speed -- and plots peak
lateral excursion vs start offset y0 for every model, fixed (dashed) vs plan-speed (solid). The track
edge is drawn as a horizontal line: a curve crossing it means the TRUE car left the track. A hard
constraint is only enforceable if the model predicts the boundary accurately, so this also separates the
physical-state models (ego/grounded) from JEPA.

    run_track_mpc_wm.py @M --hard-bound --sweep y0 --sweep-values ... --name edge_fixed
    run_track_mpc_wm.py @M --hard-bound --plan-speed --v-ref 0.20 --sweep y0 --sweep-values ... --name edge_plan
    python analysis/constraint_test.py --fixed edge_fixed --plan edge_plan --half 0.20
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import csv
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
MODEL_COLOR = {"ego": "#2a78d6", "grounded": "#1baf7a", "jepa": "#eb6834", "oracle": "#8a8781"}
MODEL_ORDER = ["oracle", "ego", "grounded", "jepa"]


def short(label):
    l = label.lower()
    return "oracle" if ("oracle" in l or "unicycle" in l) else "jepa" if "jepa" in l else "grounded" if "grounded" in l else "ego"


def read_sweep(name):
    """results/track_mpc/<name>/sweep_y0.csv -> {short_model: [(y0, max_excursion, in_bounds)]}"""
    path = ROOT / "results" / "track_mpc" / name / "sweep_y0.csv"
    if not path.exists():
        raise SystemExit(f"missing {path} — run the y0 sweep with --name {name} first")
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out.setdefault(short(row["model"]), []).append(
                (float(row["y0"]), float(row["max_excursion"]), int(row["in_bounds"])))
    for k in out:
        out[k].sort()
    return out


def main():
    ap = argparse.ArgumentParser(description="Overlay the near-edge hard-constraint test (fixed vs plan-speed).")
    ap.add_argument("--fixed", default="edge_fixed", help="--name of the fixed-speed y0 sweep")
    ap.add_argument("--plan", default="edge_plan", help="--name of the plan-speed y0 sweep")
    ap.add_argument("--half", type=float, default=0.20, help="track half-width (the boundary |y| <= half)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    fixed, plan = read_sweep(a.fixed), read_sweep(a.plan)
    out = Path(a.out) if a.out else ROOT / "results" / "track_mpc" / "constraint_test.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    models = [m for m in MODEL_ORDER if m in fixed or m in plan]

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.axhline(a.half, color="0.25", lw=1.5, ls="-", label="track edge")
    for src, style, tag in [(plan, "-", "plan-speed"), (fixed, "--", "fixed speed")]:
        for m in models:
            if m not in src:
                continue
            xs = [p[0] for p in src[m]]; ex = [p[1] for p in src[m]]
            ax.plot(xs, ex, style, marker="o", ms=4, color=MODEL_COLOR[m], label=f"{m} · {tag}")
            for x, e, ib in src[m]:            # ring the out-of-bounds points
                if not ib:
                    ax.plot(x, e, "o", ms=10, mfc="none", mec="#c0392b", mew=1.6, zorder=5)

    ax.set_xlabel("start offset  y0  (distance from centerline; edge at %.2f)" % a.half)
    ax.set_ylabel("peak |y| excursion  (crosses edge = left the track)")
    ax.set_title("Speed as a constraint DOF: plan-speed (solid) vs fixed (dashed), hard bound")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}   (red rings = out of bounds)")

    csv_out = out.with_suffix(".csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["y0", "model", "mode", "max_excursion", "in_bounds"])
        for src, mode in [(fixed, "fixed"), (plan, "plan-speed")]:
            for m, pts in src.items():
                for y0, ex, ib in pts:
                    w.writerow([f"{y0:g}", m, mode, f"{ex:.4f}", ib])
    print(f"wrote {csv_out}")


if __name__ == "__main__":
    main()
