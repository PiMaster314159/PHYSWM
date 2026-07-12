"""Slide-ready comparison figures from the unified metrics. Reads results/_compare/master_metrics.csv
(+ each run's pose_dim_corr.csv / train_history.csv / the MPC mpc_summary.csv) and writes PNGs to
results/_compare/figures/.  Run after build_master_metrics.py:  python analysis/figures.py

Figures: (1) control quality (MPC final lateral), (2) heading accuracy (angular error),
(3) grounding heatmap (pose x latent-dim correlation), (4) training curves.
Colors are a validated colorblind-safe categorical palette, fixed per model (never cycled)."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import csv
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import config as C

# --- dataviz style: white surface, recessive grid/axes, muted ink ---
plt.rcParams.update({
    "figure.facecolor": "white", "savefig.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8, "axes.grid": True, "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "text.color": "#0b0b0b", "axes.titlecolor": "#0b0b0b", "axes.labelcolor": "#52514e",
    "xtick.color": "#898781", "ytick.color": "#898781", "font.family": "sans-serif", "font.size": 11,
})
# fixed hue per entity (validated palette): physical models cool, the abstract JEPA warm/distinct
MODEL_COLOR = {"unicycle": "#898781", "ego": "#2a78d6", "grounded": "#1baf7a", "jepa": "#eb6834"}
MODEL_ORDER = ["unicycle", "ego", "grounded", "jepa"]
LABEL = {"unicycle": "unicycle (oracle)", "ego": "ego", "grounded": "grounded", "jepa": "jepa"}
DIVERGING = LinearSegmentedColormap.from_list("bwr", ["#2a78d6", "#f0efec", "#e34948"])  # blue-gray-red

OUT = C.RESULTS_DIR / "_compare" / "figures"


def _short(label):
    l = label.lower()
    return ("unicycle" if "oracle" in l or "unicycle" in l else
            "jepa" if "jepa" in l else "grounded" if "grounded" in l else "ego" if "ego" in l else label)


def load_master():
    p = C.RESULTS_DIR / "_compare" / "master_metrics.csv"
    return list(csv.DictReader(open(p))) if p.exists() else []


def _order(models):
    return sorted(models, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)


def bar(pairs, title, xlabel, fname, fmt="{:.3f}"):
    """pairs: list of (model, value). Horizontal bars, fixed colors, direct value labels."""
    pairs = [(m, float(v)) for m, v in pairs if v not in (None, "")]
    if not pairs:
        return
    pairs = sorted(pairs, key=lambda p: MODEL_ORDER.index(p[0]) if p[0] in MODEL_ORDER else 99)
    models, vals = zip(*pairs)
    fig, ax = plt.subplots(figsize=(6.2, 0.6 * len(models) + 1.4))
    y = np.arange(len(models))[::-1]
    ax.barh(y, vals, height=0.62, color=[MODEL_COLOR.get(m, "#898781") for m in models], zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([LABEL.get(m, m) for m in models])
    ax.set_xlabel(xlabel); ax.set_title(title, loc="left", pad=10, fontweight="bold")
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.015, yi, fmt.format(v), va="center", ha="left",
                color="#52514e", fontsize=10)
    ax.set_xlim(0, max(vals) * 1.18)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=150); plt.close(fig)
    print("wrote", OUT / fname)


def heatmap(rows):
    """Per-model pose x latent-dim(0..3) correlation. Diagonal = pose grounded in the block."""
    have = [r for r in rows if (C.RESULTS_DIR / r["run"] / r["model"] / r["tag"] / "pose_dim_corr.csv").exists()]
    if not have:
        return
    have = sorted(have, key=lambda r: MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 99)
    fig, axes = plt.subplots(1, len(have), figsize=(3.1 * len(have), 3.3), squeeze=False)
    for ax, r in zip(axes[0], have):
        rd = C.RESULTS_DIR / r["run"] / r["model"] / r["tag"]
        rr = list(csv.reader(open(rd / "pose_dim_corr.csv")))
        header, body = rr[0][1:], rr[1:]
        M = np.array([[float(x) for x in row[1:]] for row in body])
        poses = [row[0] for row in body]
        im = ax.imshow(M, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(header))); ax.set_xticklabels(header)
        ax.set_yticks(range(len(poses))); ax.set_yticklabels(poses)
        ax.set_title(LABEL.get(r["model"], r["model"]), fontweight="bold", color=MODEL_COLOR.get(r["model"]))
        ax.grid(False)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                        color="#0b0b0b" if abs(M[i, j]) < 0.6 else "white")
    fig.suptitle("Pose  x  latent dims 0-3  (diagonal = pose grounded in the block)",
                 x=0.02, ha="left", fontweight="bold")
    fig.colorbar(im, ax=axes[0], fraction=0.03, pad=0.02, label="correlation")
    fig.savefig(OUT / "grounding_heatmap.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", OUT / "grounding_heatmap.png")


def training(rows):
    """Total training loss over steps, one line per model (from train_history.csv)."""
    have = [r for r in rows if (C.RESULTS_DIR / r["run"] / r["model"] / r["tag"] / "train_history.csv").exists()]
    if not have:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4))
    for r in _order([h["model"] for h in have]):
        row = next(h for h in have if h["model"] == r)
        rd = C.RESULTS_DIR / row["run"] / row["model"] / row["tag"]
        hist = list(csv.DictReader(open(rd / "train_history.csv")))
        steps = [float(h["step"]) for h in hist if h.get("total")]
        total = [float(h["total"]) for h in hist if h.get("total")]
        if not steps:
            continue
        ax.plot(steps, total, lw=2, color=MODEL_COLOR.get(r, "#898781"), label=LABEL.get(r, r))
        ax.text(steps[-1], total[-1], "  " + LABEL.get(r, r), color=MODEL_COLOR.get(r, "#898781"),
                va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("training step"); ax.set_ylabel("total loss")
    ax.set_title("Training history", loc="left", pad=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "training_curves.png", dpi=150); plt.close(fig)
    print("wrote", OUT / "training_curves.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_master()
    if not rows:
        print("no master_metrics.csv — run analysis/build_master_metrics.py first."); return

    # (1) control quality — prefer the MPC summary (has the oracle too), else master rows
    mpc_csv = C.RESULTS_DIR / "track_mpc" / "mpc_summary.csv"
    if mpc_csv.exists():
        mpc = [(_short(r["model"]), r["final_lat"]) for r in csv.DictReader(open(mpc_csv))]
    else:
        mpc = [(r["model"], r.get("mpc_final_lat")) for r in rows]
    bar(mpc, "Control quality — MPC final lateral error (lower is better)",
        "final |y - centerline|  (world units)", "mpc_final_lat.png")

    # (2) heading accuracy — angular error of decode_pose
    bar([(r["model"], r.get("theta_mae_deg")) for r in rows],
        "Heading accuracy — decode_pose angular error (lower is better)",
        "mean heading error (deg)", "heading_theta_mae.png", fmt="{:.2f}")

    heatmap(rows)      # (3) grounding
    training(rows)     # (4) training history


if __name__ == "__main__":
    main()
