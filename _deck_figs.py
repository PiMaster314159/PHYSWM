"""Generate the deck figures from the probe CSVs + checkpoints. Saves to results/_deck/."""
import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
torch.set_num_threads(4)

from models.jepa import JEPA
from models.dataset import make_dataloaders
from eval.probe import extract_latents, make_mlp_probe, train_probe, state_to_target
from sim.render import render_frame

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
CK = os.path.join(ROOT, "models", "checkpoints")
DS = os.path.join(ROOT, "data", "datasets")
OUT = os.path.join(RES, "_deck")
os.makedirs(OUT, exist_ok=True)
SEED = 0


# ---- Figure 0: what the model sees (a short trajectory of frames) ----
# integrate a simple unicycle arc and render the binary frames
x, y, th = 0.30, 0.35, 0.4
v, omega, dt = 0.18, 0.7, 0.1
states = []
for _ in range(40):
    states.append((x, y, th))
    x += dt * v * np.cos(th); y += dt * v * np.sin(th); th += dt * omega
pick = np.linspace(0, len(states) - 1, 5).astype(int)
fig, axes = plt.subplots(1, 5, figsize=(13, 2.9))
for ax, i in zip(axes, pick):
    ax.imshow(render_frame(states[i], grid_size=64), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"t = {i}", fontsize=10)
fig.suptitle("What the model sees: a triangle robot driving (64 x 64 binary pixels, heading = where the nose points)",
             y=1.04, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig0_world.png"), dpi=140, bbox_inches="tight")
print("wrote fig0_world.png")
plt.close(fig)


def flip_of(run: str) -> float:
    """Read mlp-probe theta_flip_pct from a run's CSV."""
    with open(os.path.join(RES, run, "probe_metrics.csv")) as f:
        for r in csv.DictReader(f):
            if r["probe"] == "mlp":
                return float(r["theta_flip_pct"])
    raise ValueError(run)


# ---- Figure 1: the 2x2 interaction (visible x demanded), flip rate ----
binary_s1 = flip_of("run02_64x64_residual_s1_e10")
binary_s4 = flip_of("run02_64x64_residual_s4_e10")
nose_s1   = flip_of("run04_64x64_nose_residual_s1_e10")
nose_s4   = flip_of("run04_64x64_nose_residual_s4_e10")

fig, ax = plt.subplots(figsize=(7.5, 5))
groups = ["Single-step (s1)\nheading barely matters", "Multi-step (s4)\nheading matters"]
xpos = np.arange(2)
w = 0.36
b_binary = ax.bar(xpos - w/2, [binary_s1, binary_s4], w, label="Binary triangle (aliased)", color="#b0b0b0")
b_nose   = ax.bar(xpos + w/2, [nose_s1, nose_s4],   w, label="Nose dot (heading visible)", color="#2a7fb8")
ax.axhline(50, ls=":", c="gray", lw=1)
ax.text(1.45, 50.8, "chance (50%)", fontsize=8, color="gray")
for bars in (b_binary, b_nose):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.7,
                f"{bar.get_height():.0f}%", ha="center", fontsize=10)
ax.set_xticks(xpos); ax.set_xticklabels(groups)
ax.set_ylabel("heading flips  (% of frames > 90 deg wrong)")
ax.set_title("Heading is learned only when it is BOTH visible AND task-relevant")
ax.set_ylim(0, 55)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_interaction.png"), dpi=140)
print("wrote fig1_interaction.png  ", binary_s1, binary_s4, nose_s1, nose_s4)
plt.close(fig)


# ---- Figure 2: dose-response (64px, nose, residual): flip vs horizon ----
steps = [1, 2, 4]
flips = [flip_of("run04_64x64_nose_residual_s1_e10"),
         flip_of("run04_64x64_nose_residual_s2_e10"),
         flip_of("run04_64x64_nose_residual_s4_e10")]
fig, ax = plt.subplots(figsize=(7.0, 5))
ax.plot(steps, flips, "o-", color="#2a7fb8", lw=2, ms=9)
for s, fl in zip(steps, flips):
    ax.annotate(f"{fl:.0f}%", (s, fl), textcoords="offset points", xytext=(0, 10), fontsize=11)
ax.set_xticks(steps)
ax.set_xlabel("prediction horizon  (steps between frame_t and frame_t+k)")
ax.set_ylabel("heading flips  (% of frames > 90 deg wrong)")
ax.set_title("Dose-response: more task demand for heading -> fewer flips\n(64px, nose dot, no physics prior)")
ax.set_ylim(0, max(flips) * 1.25)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_dose_response.png"), dpi=140)
print("wrote fig2_dose_response.png  ", flips)
plt.close(fig)


# ---- Figure 3: heading-error histogram, before vs after ----
def heading_errors(ckpt, h5, grid):
    """Per-frame heading error (deg) on val, MLP probe trained on train split."""
    m = JEPA(grid_size=grid, latent_dim=128, predictor_mode="residual")
    m.load_state_dict(torch.load(os.path.join(CK, ckpt), map_location="cpu"))
    train_dl, val_dl = make_dataloaders(os.path.join(DS, h5), batch_size=256, seed=SEED,
                                        return_state=True, step=1)
    Z_tr, S_tr = extract_latents(m, train_dl, "cpu")
    Z_va, S_va = extract_latents(m, val_dl, "cpu")
    torch.manual_seed(SEED)
    probe = train_probe(make_mlp_probe(Z_tr.shape[1]), Z_tr, state_to_target(S_tr),
                        epochs=40, device="cpu")
    with torch.no_grad():
        pred = probe(Z_va).numpy()
    d = np.arctan2(pred[:, 3], pred[:, 2]) - S_va.numpy()[:, 2]
    return np.degrees(np.abs(np.arctan2(np.sin(d), np.cos(d))))

err_before = heading_errors("run02_64x64_residual_s1_e10.pt", "run02_64x64.h5", 64)
err_after  = heading_errors("run04_64x64_nose_residual_s2_e10.pt", "run04_64x64_nose.h5", 64)

fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(0, 180, 37)
ax.hist(err_before, bins=bins, density=True, color="#b0b0b0", alpha=0.85,
        label=f"Binary, single-step  ({(err_before>90).mean()*100:.0f}% flipped)")
ax.hist(err_after, bins=bins, density=True, color="#2a7fb8", alpha=0.7,
        label=f"Nose dot, multi-step  ({(err_after>90).mean()*100:.0f}% flipped)")
ax.axvline(90, ls="--", c="k", lw=1.2)
ax.text(94, 0.019, "flip threshold:\n> 90 deg = backwards", fontsize=9, color="#333333")
ax.annotate("front/back\nflip tail", xy=(165, 0.007), xytext=(120, 0.016),
            fontsize=9, color="#555555", ha="center",
            arrowprops=dict(arrowstyle="->", color="#888888"))
ax.set_xlabel("heading error (deg)")
ax.set_ylabel("density of frames")
ax.set_title("The front/back flip tail collapses when heading is visible AND demanded")
ax.set_xlim(0, 180); ax.set_xticks(range(0, 181, 30))
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_heading_histogram.png"), dpi=140)
print("wrote fig3_heading_histogram.png")
plt.close(fig)

print("deck figs done")
