"""Physics-focused figure: does embedding the kinematic prior help? (reads CSVs only)."""
import os, csv
import numpy as np
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT = os.path.join(RES, "_deck")


def flip_of(run):
    with open(os.path.join(RES, run, "probe_metrics.csv")) as f:
        for r in csv.DictReader(f):
            if r["probe"] == "mlp":
                return float(r["theta_flip_pct"])


# both with the nose dot (heading visible); residual = no physics, physics = kinematic prior
resid = [flip_of("run04_64x64_nose_residual_s1_e10"),
         flip_of("run04_64x64_nose_residual_s2_e10"),
         flip_of("run04_64x64_nose_residual_s4_e10")]
phys  = [flip_of("run04_64x64_nose_physics_s1_e10"),
         flip_of("run04_64x64_nose_physics_s2_e10"),
         flip_of("run04_64x64_nose_physics_s4_e10")]

x = np.arange(3)
w = 0.36
fig, ax = plt.subplots(figsize=(8.2, 5))
b1 = ax.bar(x - w/2, resid, w, label="No physics (plain predictor)", color="#9AA3AD")
b2 = ax.bar(x + w/2, phys,  w, label="Physics prior (embedded kinematics)", color="#1C7293")
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{bar.get_height():.0f}%",
                ha="center", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(["1 step\n(weak demand)", "2 steps", "4 steps\n(strong demand)"])
ax.set_xlabel("prediction horizon (how much the task demands heading)")
ax.set_ylabel("heading flips  (% of frames > 90 deg wrong)")
ax.set_title("Where embedding physics helps: it substitutes for task demand\n(both runs use the nose dot, so heading is visible)")
ax.set_ylim(0, 32)
ax.legend(loc="upper right")
# annotate the win
ax.annotate("physics alone\ncracks heading\nat 1 step",
            xy=(0.18, 10), xytext=(0.6, 20), fontsize=10, color="#1C7293",
            arrowprops=dict(arrowstyle="->", color="#1C7293"))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_physics_vs_nophysics.png"), dpi=140)
print("wrote fig4_physics_vs_nophysics.png  resid=", resid, " phys=", phys)
