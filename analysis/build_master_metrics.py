"""Assemble results/_compare/master_metrics.csv from every run's metrics.json — one row per
(run, model, tag), pose + dynamics + coeffs + MPC side by side. Run after training + the MPC
comparison:  python analysis/build_master_metrics.py"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import csv
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as C
from eval.metrics import collect

# preferred column order (any extra keys are appended)
ORDER = ["run", "model", "tag",
         "theta_mae_deg", "theta_flip_pct", "x_r2", "y_r2", "pos_rmse",
         "dynamics_pred_pos_err", "dynamics_pred_theta_mae_deg",
         "coeffs_a_v", "coeffs_a_omega",
         "mpc_final_lat", "mpc_final_head_deg", "mpc_max_excursion", "mpc_rms_lat", "mpc_in_bounds"]


def main():
    rows = collect(C.RESULTS_DIR)
    if not rows:
        print(f"no metrics.json found under {C.RESULTS_DIR} — train a model first.")
        return
    keys = list(dict.fromkeys([k for k in ORDER if any(k in r for r in rows)]
                              + [k for r in rows for k in r]))
    out_dir = C.RESULTS_DIR / "_compare"; out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "master_metrics.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(keys)
        for r in rows:
            w.writerow([r.get(k, "") for k in keys])
    print(f"wrote {out}  ({len(rows)} runs)")
    show = [k for k in ("model", "tag", "theta_mae_deg", "pos_rmse", "mpc_final_lat") if k in keys]
    print("  " + "  ".join(f"{s:>14}" for s in show))
    for r in rows:
        print("  " + "  ".join(f"{str(r.get(s, '')):>14.14}" for s in show))


if __name__ == "__main__":
    main()
