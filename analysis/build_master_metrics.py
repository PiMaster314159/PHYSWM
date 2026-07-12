"""Assemble results/_compare/master_metrics.csv from the per-run eval CSVs.

Pure CSV parsing (no torch/h5py), so it runs anywhere. Re-run after new training runs to
refresh the summary table. Reads each run0*/ result folder and pulls pose recovery (mlp +
linear probe), the ego direct readout, the actuator a_v recovery, and the 1-step prediction.
"""
import csv
import re
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
RES = ROOT / "results"


def read_keyed(path, key="probe"):
    with open(path) as f:
        return {row[key]: row for row in csv.DictReader(f)}


def read_one(path):
    with open(path) as f:
        return next(csv.DictReader(f))


def parse(name):
    for tok, model in (("_ego_", "ego"), ("_residual_", "residual"), ("_grounded_", "grounded")):
        if tok in name:
            run, tag = name.split(tok)
            step = int(re.match(r"s(\d+)", tag).group(1))
            if model == "residual":
                variant = "readout" if "_rd" in f"_{tag}" else "anchor"
            elif model == "ego":
                variant = "learnable" if "learn" in tag else "frozen"
            else:
                variant = "block-anchor"
            return run, model, variant, step
    return None


COLS = ["run", "model", "pred_step", "mlp_theta_mae", "mlp_x_r2", "mlp_y_r2",
        "lin_theta_mae", "direct_theta_mae", "a_v_learned", "true_gain",
        "a_v/true (bias)", "predict_pos_err", "predict_theta_mae"]

rows = []
for d in sorted(RES.iterdir()):
    if not (d.is_dir() and re.match(r"run0[78]_", d.name)):
        continue
    p = parse(d.name)
    if not p:
        continue
    run, model, variant, step = p
    r = {c: "" for c in COLS}
    r.update(run=run, model=f"{model} ({variant})", pred_step=step)

    if model == "ego":
        sp = read_keyed(d / "state_probe.csv")
        r["mlp_theta_mae"] = sp["mlp"]["theta_mae_deg"]; r["mlp_x_r2"] = sp["mlp"]["x_r2"]; r["mlp_y_r2"] = sp["mlp"]["y_r2"]
        r["lin_theta_mae"] = sp["linear"]["theta_mae_deg"]
        if (d / "state_direct.csv").exists():
            r["direct_theta_mae"] = read_one(d / "state_direct.csv")["theta_mae_deg"]
        if (d / "actuator_recovery.csv").exists():
            ar = read_one(d / "actuator_recovery.csv")
            r["a_v_learned"] = ar["learned_a_v"]; r["true_gain"] = ar["true_gain"]
            r["a_v/true (bias)"] = f"{float(ar['learned_a_v']) / float(ar['true_gain']):.3f}"
        if (d / "predict_eval.csv").exists():
            pv = read_one(d / "predict_eval.csv")
            r["predict_pos_err"] = pv["pred_pos_err"]; r["predict_theta_mae"] = pv["pred_theta_mae_deg"]
    else:
        pm = read_keyed(d / "probe_metrics.csv")
        r["mlp_theta_mae"] = pm["mlp"]["theta_mae_deg"]; r["mlp_x_r2"] = pm["mlp"]["x_r2"]; r["mlp_y_r2"] = pm["mlp"]["y_r2"]
        r["lin_theta_mae"] = pm["linear"]["theta_mae_deg"]
    rows.append(r)

rows.sort(key=lambda r: (r["run"], r["model"], r["pred_step"]))
out = RES / "_compare" / "master_metrics.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
print(f"wrote {out}  ({len(rows)} runs)")
