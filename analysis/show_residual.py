"""Print the recovered gray-box residual law from a trained ego/grounded checkpoint — no retraining.
Runs on any machine (CPU). Grounded's coefficients depend on the free latent, so it needs --run to
sample one from the dataset; ego's are global and need no data.

    python analysis/show_residual.py --model ego --ckpt models/checkpoints/64x64_drag/ego/<tag>.pt
    python analysis/show_residual.py --model grounded --run 64x64_drag \
        --ckpt models/checkpoints/64x64_drag/grounded/<tag>.pt
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import argparse
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import config as C
from models.components import format_residual


def main():
    p = argparse.ArgumentParser(description="Decode a trained structured residual into named physics terms.")
    p.add_argument("--model", required=True, choices=["ego", "grounded"])
    p.add_argument("--ckpt", required=True, help="path to the .pt checkpoint (trained with --residual basis)")
    p.add_argument("--run", default=None, help="dataset stem; grounded needs it for a free latent")
    p.add_argument("--grid-size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    if a.model == "ego":
        from models.state_ae import EgoWorldModel
        m = EgoWorldModel(grid_size=a.grid_size, learn_coeffs=True, residual_mode="basis")
    else:
        from models.grounded import GroundedJEPA
        m = GroundedJEPA(grid_size=a.grid_size, latent_dim=C.LATENT_DIM, block_dim=4,
                         learn_coeffs=True, residual_mode="basis")
    m.load_state_dict(torch.load(a.ckpt, map_location=a.device, weights_only=True), strict=False)
    m = m.to(a.device).eval()

    res = m.residual if a.model == "ego" else m.predictor.residual
    if res is None:
        raise SystemExit("this checkpoint has no structured residual (was it trained with --residual basis?)")

    free = None
    if a.model == "grounded":
        if a.run is None:
            raise SystemExit("grounded's coefficients depend on the free latent — pass --run <dataset>")
        from models.dataset import make_dataloaders
        _, va = make_dataloaders(C.DATASETS_DIR / f"{a.run}.h5", batch_size=256)
        with torch.no_grad():
            z = m.encode(next(iter(va))["frame"].to(a.device))
        free = z[:, m.block_dim:].mean(0, keepdim=True)

    a_v = (m.log_a_v if a.model == "ego" else m.predictor.log_a_v).exp().item()
    print(format_residual(res, free))
    print(f"\n(constant gray-box a_v = {a_v:.4f})")


if __name__ == "__main__":
    main()
