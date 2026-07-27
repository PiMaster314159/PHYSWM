"""Overlay the v_ref extrapolation sweep: does structured g (basis) hold as the commanded speed
leaves the training band, while the free-form MLP drifts?

Reads the two speed-planning sweeps produced by:
    run_track_mpc_wm.py ... --residual basis --plan-speed --sweep vref --name <basis-name>
    run_track_mpc_wm.py ... --residual mlp   --plan-speed --sweep vref --name <mlp-name>
and plots speed error vs v_ref for ego + grounded, basis (solid) vs mlp (dashed), oracle for scale.
A secondary top axis shows the COMMANDED speed v (via the drag curve), so you can see when the plan
leaves the training band. Optionally pass --train-v-max to mark that boundary.

    python analysis/vref_extrap.py --basis drag_vref_basis --mlp drag_vref_mlp --gain 1 --drag-c 1
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import csv
import math
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
MODEL_COLOR = {"ego": "#2a78d6", "grounded": "#1baf7a", "oracle": "#8a8781"}


def short(label):
    l = label.lower()
    return "oracle" if ("oracle" in l or "unicycle" in l) else "jepa" if "jepa" in l else "grounded" if "grounded" in l else "ego"


def read_sweep(name):
    """results/track_mpc/<name>/sweep_vref.csv -> {short_model: [(v_ref, speed_err), ...]}"""
    path = ROOT / "results" / "track_mpc" / name / "sweep_vref.csv"
    if not path.exists():
        raise SystemExit(f"missing {path} — run the vref sweep with --name {name} first")
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out.setdefault(short(row["model"]), []).append((float(row["vref"]), float(row["speed_err"])))
    for k in out:
        out[k].sort()
    return out


def main():
    ap = argparse.ArgumentParser(description="Overlay the v_ref extrapolation sweep (basis vs mlp).")
    ap.add_argument("--basis", default="drag_vref_basis", help="--name of the basis sweep run")
    ap.add_argument("--mlp", default="drag_vref_mlp", help="--name of the mlp sweep run")
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--drag-c", type=float, default=1.0)
    ap.add_argument("--train-v-max", type=float, default=None,
                    help="training band's max commanded v; draws the extrapolation boundary")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    basis, mlp = read_sweep(a.basis), read_sweep(a.mlp)
    out = Path(a.out) if a.out else ROOT / "results" / "track_mpc" / "vref_extrap.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    # v_ref -> commanded v (lower branch of gain*v - drag_c*v^2 = v_ref) and its inverse, for the top axis
    g, c = a.gain, a.drag_c
    v_peak = g * g / (4 * c)                     # max feasible ground speed (drag-curve peak)
    def vref_to_v(vr):
        vr = np.clip(np.asarray(vr, float), 0, v_peak - 1e-9)
        return (g - np.sqrt(np.maximum(g * g - 4 * c * vr, 0.0))) / (2 * c)
    def v_to_vref(v):
        v = np.asarray(v, float)
        return g * v - c * v * v

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for src, style, tag in [(basis, "-", "basis"), (mlp, "--", "mlp")]:
        for model in ("ego", "grounded"):
            if model not in src:
                continue
            xs = [p[0] for p in src[model]]; ys = [p[1] for p in src[model]]
            ax.plot(xs, ys, style, marker="o", ms=4, color=MODEL_COLOR[model], label=f"{model} · {tag}")
    if "oracle" in basis:                        # oracle is mode-independent; plot once for scale
        xs = [p[0] for p in basis["oracle"]]; ys = [p[1] for p in basis["oracle"]]
        ax.plot(xs, ys, ":", color=MODEL_COLOR["oracle"], lw=1.5, label="oracle (naive)")

    if a.train_v_max is not None:                # commanded v beyond here = extrapolation
        vr_edge = float(v_to_vref(a.train_v_max))
        ax.axvline(vr_edge, color="0.4", lw=1.2, ls="-.")
        ax.text(vr_edge, ax.get_ylim()[1] * 0.96, f"  train edge (v={a.train_v_max:g})",
                color="0.35", fontsize=9, va="top")

    ax.set_xlabel("target ground speed  v_ref"); ax.set_ylabel("speed error  |mean_v − v_ref|  (lower = better)")
    ax.set_title("Extrapolation: structured g (solid) vs free-form MLP (dashed)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, ncol=2)
    secax = ax.secondary_xaxis("top", functions=(vref_to_v, v_to_vref))
    secax.set_xlabel("commanded speed  v  (via drag curve)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")

    # merged CSV for the record
    csv_out = out.with_suffix(".csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["v_ref", "commanded_v", "model", "mode", "speed_err"])
        for src, mode in [(basis, "basis"), (mlp, "mlp")]:
            for model, pts in src.items():
                for vr, se in pts:
                    w.writerow([f"{vr:g}", f"{float(vref_to_v(vr)):.4f}", model, mode, f"{se:.4f}"])
    print(f"wrote {csv_out}")


if __name__ == "__main__":
    main()
