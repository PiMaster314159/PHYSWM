import os
import csv

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

rows = []
for name in sorted(os.listdir(RES)):
    csv_path = os.path.join(RES, name, "probe_metrics.csv")
    if not os.path.isfile(csv_path):
        continue
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["probe"] == "mlp":
                rows.append((name, r))

cols = ["pos_rmse", "theta_mae_deg", "theta_median_deg", "theta_flip_pct"]
print(f"{'run (mlp probe)':38s} {'pos_rmse':>9s} {'mae':>7s} {'median':>7s} {'flip%':>7s}")
print("-" * 74)
for name, r in rows:
    has_flip = "theta_flip_pct" in r and r["theta_flip_pct"] not in (None, "")
    flip = f"{float(r['theta_flip_pct']):7.1f}" if has_flip else "    n/a"
    print(f"{name:38s} {float(r['pos_rmse']):9.4f} {float(r['theta_mae_deg']):7.1f} "
          f"{float(r['theta_median_deg']):7.1f} {flip}")
