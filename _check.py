import json, glob, os, csv
print("=== budget re-run: recovered v^2 coefficient ===")
found = False
for p in sorted(glob.glob("results/64x64_bicycle_drag5/*/*/metrics.json")):
    tag = os.path.basename(os.path.dirname(p))
    j = json.load(open(p))
    c = j.get("residual_coeffs")
    if c:
        found = True
        v2 = c["velocity"].get("v^2")
        note = "b04" if "b04" in tag else "budget 0.1"
        print(f"  {j['model']:9s} {note:10s} v^2 = {v2:+.3f}   (target -5.0)   v_rmse={j['velocity']['v_rmse']:.4f}  pos={j['pose']['pos_rmse']:.4f}")
if not found:
    print("  (none found)")

print("\n=== track_mpc result folders ===")
for d in sorted(glob.glob("results/track_mpc/*/")):
    print("  ", d)

for s in sorted(glob.glob("results/track_mpc/*/mpc_summary.csv")):
    print(f"\n=== {s} ===")
    for row in csv.DictReader(open(s)):
        print("  " + "  ".join(f"{k}={v}" for k, v in row.items()))
