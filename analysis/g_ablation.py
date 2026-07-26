"""Collect the g-ablation into one table + figure: {ego, grounded} × {none, basis, mlp}.

Reads pose RMSE (training eval) and speed_err (MPC with --plan-speed) from each checkpoint's
results/<run>/<model>/<tag>/metrics.json -- the MPC harness merges its control result into that same
file via update_mpc, so both numbers live in one place. The residual mode is inferred from the tag
suffix (_gbasis / _gmlp / none), which the ablation training runs produce.

    python analysis/g_ablation.py --run 64x64_drag

The two panels answer the thesis directly: does 'mlp' match 'basis' on FIT (pose RMSE) while
diverging on CONTROL (speed err)? If so, structure buys control, not fit.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import json
import csv
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
MODES = ["none", "basis", "mlp"]
MODE_LABEL = {"none": "none\n(kinematics)", "basis": "basis\n(structured g)", "mlp": "mlp\n(free-form)"}
MODEL_COLOR = {"ego": "#2a78d6", "grounded": "#1baf7a"}


def infer_mode(tag):
    if "_gbasis" in tag:
        return "basis"
    if "_gmlp" in tag:
        return "mlp"
    return "none"


def collect(run, models, exclude_learn):
    """(model, mode) -> {pose_rmse, speed_err, mean_speed, final_lat, tag}. First match per cell wins."""
    rows = {}
    for model in models:
        d = ROOT / "results" / run / model
        if not d.exists():
            continue
        for mj in sorted(d.glob("*/metrics.json")):
            tag = mj.parent.name
            if exclude_learn and "_learn" in tag:      # old --learn-coeffs runs predate the mode-tag; skip
                continue
            mode = infer_mode(tag)
            key = (model, mode)
            if key in rows:
                print(f"  [warn] extra {model}/{mode} checkpoint '{tag}' ignored (using '{rows[key]['tag']}')")
                continue
            m = json.load(open(mj))
            mpc = m.get("mpc", {})
            rows[key] = {"tag": tag,
                         "pose_rmse": m.get("pose", {}).get("pos_rmse"),
                         "speed_err": mpc.get("speed_err"),
                         "mean_speed": mpc.get("mean_speed"),
                         "final_lat": mpc.get("final_lat")}
    return rows


def fmt(x, d=4):
    return "—" if x is None else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser(description="Collect the g-ablation table + figure.")
    ap.add_argument("--run", default="64x64_drag", help="dataset stem the ablation was trained on")
    ap.add_argument("--models", default="ego,grounded")
    ap.add_argument("--out", default=None, help="output dir (default results/<run>/_g_ablation)")
    ap.add_argument("--include-learn", action="store_true",
                    help="also include --learn-coeffs checkpoints (default: ablation set only; note their "
                         "tags predate the mode suffix, so mode inference may be wrong)")
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    out = Path(a.out) if a.out else ROOT / "results" / a.run / "_g_ablation"
    out.mkdir(parents=True, exist_ok=True)

    rows = collect(a.run, models, exclude_learn=not a.include_learn)
    if not rows:
        raise SystemExit(f"no metrics.json found under results/{a.run}/{{{','.join(models)}}}/ "
                         f"(train the ablation first; add --include-learn to pick up older runs)")

    # ---- table -> stdout + CSV ----
    print(f"\n=== g-ablation: {a.run} ===  (pose RMSE = fit, speed err = control)")
    hdr = f"{'model':9s} {'mode':6s} {'pose RMSE':>10s} {'speed err':>10s} {'mean v':>8s} {'final lat':>10s}  tag"
    print(hdr); print("-" * len(hdr))
    csv_path = out / "g_ablation.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "mode", "pose_rmse", "speed_err", "mean_speed", "final_lat", "tag"])
        for model in models:
            for mode in MODES:
                r = rows.get((model, mode))
                if r is None:
                    print(f"{model:9s} {mode:6s} {'—':>10s} {'—':>10s} {'—':>8s} {'—':>10s}  (missing)")
                    w.writerow([model, mode, "", "", "", "", ""])
                    continue
                print(f"{model:9s} {mode:6s} {fmt(r['pose_rmse']):>10s} {fmt(r['speed_err']):>10s} "
                      f"{fmt(r['mean_speed'], 3):>8s} {fmt(r['final_lat']):>10s}  {r['tag']}")
                w.writerow([model, mode, fmt(r['pose_rmse']), fmt(r['speed_err']),
                            fmt(r['mean_speed'], 3), fmt(r['final_lat']), r['tag']])
    print(f"\nwrote {csv_path}")

    # ---- figure: FIT vs CONTROL, grouped bars (model) over modes ----
    x = np.arange(len(MODES)); wbar = 0.36
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    panels = [(ax1, "pose_rmse", "Fit — pose RMSE", "pose RMSE  (lower = better fit)"),
              (ax2, "speed_err", "Control — speed error", "|achieved − v_ref|  (lower = better control)")]
    for ax, key, title, ylab in panels:
        for i, model in enumerate(models):
            vals = [(rows.get((model, m)) or {}).get(key) for m in MODES]
            heights = [(v if v is not None else 0.0) for v in vals]
            xoff = x + (i - (len(models) - 1) / 2) * wbar
            ax.bar(xoff, heights, wbar, color=MODEL_COLOR.get(model, "#888"), label=model)
            for xi, v in zip(xoff, vals):
                txt = "—" if v is None else f"{v:.3f}"
                ax.text(xi, (v if v is not None else 0) + 1e-4, txt, ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x); ax.set_xticklabels([MODE_LABEL[m] for m in MODES], fontsize=9)
        ax.set_title(title); ax.set_ylabel(ylab, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=9, frameon=False)
    fig.suptitle(f"g-ablation ({a.run}): does structure buy fit, or control?", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path = out / "g_ablation.png"
    fig.savefig(fig_path, dpi=150); plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
