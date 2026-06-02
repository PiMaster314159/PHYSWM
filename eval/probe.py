#!/usr/bin/env python
# coding: utf-8

# # Probing the frozen latent
# 
# Did the encoder secretly learn physical state from pixels alone? Probing measures this by training a small network on top of frozen encoder representations:
# 
# ```
# frame -> [FROZEN encoder] -> z -> [small trainable probe] -> (x, y, theta)
# ```
# 
# The encoder is never updated. If the probe achieves low error, the latent z must already encode position and heading. Two probe types: a linear probe (one matrix, tests linear readability) and an MLP probe (one hidden layer, tests whether information is present even if entangled).
# 
# > To generate `eval/probe.py` for importing, run from `PHYSWM/`:
# > ```
# > jupyter nbconvert --to python eval/probe.ipynb
# > ```

# In[ ]:


import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "constants.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_PATH, CKPT_PATH


# ## Path setup

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Union

try:
    from models.jepa import JEPA
    from models.dataset import make_dataloaders
except ImportError:
    from jepa import JEPA
    from dataset import make_dataloaders


# ## Imports

# In[ ]:


@torch.no_grad()
def extract_latents(model: JEPA, dl: DataLoader, device: str):
    """Run the frozen encoder over all batches; return (Z, states) as CPU tensors.

    Parameters
    ----------
    model : JEPA
        Set to eval mode internally.
    dl : DataLoader
        Must be constructed with return_state=True.
    device : str

    Returns
    -------
    Z : (N, latent_dim) float32
    states : (N, 3) float32 -- true (x, y, theta)
    """
    model.eval()
    zs, states = [], []
    for batch in dl:
        z = model.encode(batch["frame"].to(device))
        zs.append(z.cpu())
        states.append(batch["state"])
    return torch.cat(zs), torch.cat(states)


# ## Extract latents
# 
# Run the frozen encoder once over the full dataset and cache the results. The probe trains on these cached tensors directly, so the encoder is never called again during probe training.

# In[ ]:


def state_to_target(states: torch.Tensor) -> torch.Tensor:
    """(N, 3) (x, y, theta) -> (N, 4) (x, y, cos theta, sin theta).

    Heading is encoded as (cos, sin) to remove the wraparound seam at +/-pi.
    """
    x, y, theta = states[:, 0], states[:, 1], states[:, 2]
    return torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=1)


def target_to_state(pred: torch.Tensor) -> torch.Tensor:
    """(N, 4) probe output -> (N, 3) (x, y, theta) via atan2."""
    x, y  = pred[:, 0], pred[:, 1]
    theta = torch.atan2(pred[:, 3], pred[:, 2])
    return torch.stack([x, y, theta], dim=1)


def angular_error(theta_pred: torch.Tensor, theta_true: torch.Tensor) -> torch.Tensor:
    """Wrapped absolute angle difference in radians, elementwise.

    Maps any difference back into [0, pi] so +pi and -pi count as the same direction.
    """
    d = theta_pred - theta_true
    return torch.abs(torch.atan2(torch.sin(d), torch.cos(d)))


# ## Angle helpers
# 
# Heading is circular so the probe predicts `(cos theta, sin theta)` instead of raw theta, then `target_to_state` recovers the angle with `atan2`. `angular_error` wraps the difference into `[0, pi]` so +179 and -179 degrees correctly count as 2 degrees apart.

# In[ ]:


def make_linear_probe(latent_dim: int, out_dim: int = 4) -> nn.Linear:
    """Single linear layer: z -> (x, y, cos theta, sin theta).

    If this works well, state is linearly readable from the latent.
    """
    return nn.Linear(latent_dim, out_dim)


def make_mlp_probe(latent_dim: int, out_dim: int = 4, hidden: int = 128) -> nn.Sequential:
    """One hidden layer: tests whether state is present even if entangled nonlinearly."""
    return nn.Sequential(
        nn.Linear(latent_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
    )


# ## Probes
# 
# Two small regressors from latent to the 4-vector target `(x, y, cos theta, sin theta)`.

# In[ ]:


def train_probe(
    probe: nn.Module,
    Z_tr: torch.Tensor,
    T_tr: torch.Tensor,
    epochs: int = 40,
    lr: float = 1e-3,
    batch_size: int = 512,
    device: Optional[str] = None,
) -> nn.Module:
    """Fit a probe with Adam + MSE on precomputed latents.

    Parameters
    ----------
    probe : nn.Module
    Z_tr : (N, D) precomputed training latents
    T_tr : (N, 4) training targets from state_to_target
    epochs, lr, batch_size : training hyperparameters
    device : str, optional

    Returns
    -------
    Trained probe (same object, moved to device).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    probe.to(device).train()
    Z_tr, T_tr = Z_tr.to(device), T_tr.to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    n   = Z_tr.shape[0]
    for _ in range(epochs):
        idx        = torch.randperm(n, device=device)
        Z_tr, T_tr = Z_tr[idx], T_tr[idx]
        for i in range(0, n, batch_size):
            pred = probe(Z_tr[i:i + batch_size])
            loss = F.mse_loss(pred, T_tr[i:i + batch_size])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return probe


# ## Train probe
# 
# Standard supervised regression on the precomputed latents. The encoder is never called here.

# In[ ]:


@torch.no_grad()
def evaluate_probe(
    probe: nn.Module,
    Z: torch.Tensor,
    states: torch.Tensor,
    device: Optional[str] = None,
) -> dict:
    """Score a trained probe on held-out latents.

    Parameters
    ----------
    probe : nn.Module
    Z : (N, D) val latents from extract_latents
    states : (N, 3) true (x, y, theta) for the val set
    device : str, optional

    Returns
    -------
    dict with keys:
        x_rmse, y_rmse, pos_rmse : world units (arena is [0, 1]^2)
        theta_mae_deg, theta_median_deg : mean/median wrapped heading error in degrees
        x_r2, y_r2 : coefficient of determination (1.0 is perfect, 0.0 is chance)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    probe.to(device).eval()
    Z, states = Z.to(device), states.to(device)

    pred_states = target_to_state(probe(Z))
    err_xy      = pred_states[:, :2] - states[:, :2]
    ang         = angular_error(pred_states[:, 2], states[:, 2])
    deg         = 180.0 / torch.pi

    def r2(p: torch.Tensor, t: torch.Tensor) -> float:
        return (1 - (p - t).pow(2).sum() / (t - t.mean()).pow(2).sum()).item()

    return {
        "x_rmse":           err_xy[:, 0].pow(2).mean().sqrt().item(),
        "y_rmse":           err_xy[:, 1].pow(2).mean().sqrt().item(),
        "pos_rmse":         err_xy.pow(2).sum(1).mean().sqrt().item(),
        "theta_mae_deg":    (ang.mean()   * deg).item(),
        "theta_median_deg": (ang.median() * deg).item(),
        "x_r2":             r2(pred_states[:, 0], states[:, 0]),
        "y_r2":             r2(pred_states[:, 1], states[:, 1]),
    }


# ## Evaluate probe
# 
# Score a trained probe on held-out val latents. Reports position RMSE in world units and heading MAE in degrees. Chance level for position is ~0.29 (predicting the mean); chance level for heading is 90 degrees.

# In[ ]:


def chance_baseline(
    states_tr: torch.Tensor,
    states_val: torch.Tensor,
) -> dict:
    """Metrics for a probe that always predicts the training mean position.

    Heading chance level is 90 degrees: the expected wrapped absolute error
    of a uniformly random angle is pi/2 radians = 90 degrees.
    """
    mean_xy = states_tr[:, :2].mean(dim=0)
    err_xy  = mean_xy.unsqueeze(0) - states_val[:, :2]
    return {
        "x_rmse":           err_xy[:, 0].pow(2).mean().sqrt().item(),
        "y_rmse":           err_xy[:, 1].pow(2).mean().sqrt().item(),
        "pos_rmse":         err_xy.pow(2).sum(1).mean().sqrt().item(),
        "theta_mae_deg":    90.0,
        "theta_median_deg": 90.0,
        "x_r2":             0.0,
        "y_r2":             0.0,
    }


# ## Chance baseline
# 
# What the metrics look like for a probe that ignores the latent entirely. A real probe near these values learned nothing.

# In[ ]:


def _print_table(results: dict) -> None:
    """Print a comparison table of chance, linear, and MLP probe metrics."""
    cols = ["pos_rmse", "x_rmse", "y_rmse", "theta_mae_deg", "theta_median_deg",
            "x_r2", "y_r2"]
    head = "probe      " + "".join(f"{c:>17s}" for c in cols)
    print(head)
    print("-" * len(head))
    for name in ("chance", "linear", "mlp"):
        if name not in results:
            continue
        m = results[name]
        print(f"{name:11s}" + "".join(f"{m[c]:>17.4f}" for c in cols))


# ## Run probe
# 
# Load a checkpoint, extract frozen latents, fit both probes, and print a comparison table against the chance baseline.

# In[ ]:


def run_probe(
    ckpt_path: Optional[Union[str, Path]] = None,
    model=None,
    data_path: Union[str, Path] = DATA_PATH,
    latent_dim: int = 128,
    batch_size: int = 256,
    probe_epochs: int = 40,
    device: Optional[str] = None,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Probe a trained JEPA for (x, y, theta). Returns a results dict.

    Parameters
    ----------
    ckpt_path : str or Path, optional
        Path to a saved model checkpoint. Required when model is None.
    model : JEPA, optional
        Pre-loaded model. If provided, ckpt_path is ignored.
    data_path : str or Path
        HDF5 dataset to probe on.
    latent_dim : int
        Latent dimension. Must match the checkpoint.
    batch_size : int
    probe_epochs : int
        Training epochs for each probe.
    device : str, optional
    seed : int
        Must match the seed used during JEPA training to get the same episode split.
    verbose : bool

    Returns
    -------
    dict with keys chance, linear, mlp, each a metrics dict from evaluate_probe.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_dl, val_dl = make_dataloaders(
        data_path, batch_size=batch_size, seed=seed, return_state=True,
    )
    grid_size = train_dl.dataset.grid_size

    if model is None:
        assert ckpt_path is not None, "pass either model or ckpt_path"
        model = JEPA(grid_size=grid_size, latent_dim=latent_dim)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        if verbose:
            print(f"loaded {ckpt_path}")
    model = model.to(device)

    Z_tr, S_tr   = extract_latents(model, train_dl, device)
    Z_val, S_val = extract_latents(model, val_dl,   device)
    T_tr         = state_to_target(S_tr)
    probe_in     = Z_tr.shape[1]
    if verbose:
        print(f"latents: {Z_tr.shape[0]} train, {Z_val.shape[0]} val  (dim {probe_in})")

    results = {"chance": chance_baseline(S_tr, S_val)}
    for name, factory in [("linear", make_linear_probe), ("mlp", make_mlp_probe)]:
        torch.manual_seed(seed)
        probe = factory(probe_in)
        probe = train_probe(probe, Z_tr, T_tr, epochs=probe_epochs, device=device)
        results[name] = evaluate_probe(probe, Z_val, S_val, device=device)

    if verbose:
        _print_table(results)
    return results


# ## Tests
# 
# Runs the full probe pipeline on a random-weight encoder to verify the pipeline works. Results will be near chance since the encoder is untrained.

# In[ ]:


def _test_probe():
    if not DATA_PATH.exists():
        print(f"(skipping probe test: {DATA_PATH} not found)")
        return

    # Use a random-weight encoder -- testing the pipeline, not the results.
    train_dl, val_dl = make_dataloaders(DATA_PATH, batch_size=64, return_state=True)
    g     = train_dl.dataset.grid_size
    model = JEPA(grid_size=g, latent_dim=128)

    results = run_probe(model=model, data_path=DATA_PATH, probe_epochs=3, verbose=True)

    assert set(results.keys()) >= {"chance", "linear", "mlp"}
    for name in ("chance", "linear", "mlp"):
        assert "pos_rmse" in results[name]
        assert torch.isfinite(torch.tensor(results[name]["pos_rmse"]))

    print("All probe tests passed.")


# In[ ]:


if __name__ == "__main__":
    _test_probe()

