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

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_PATH, CKPT_PATH, SEED, REPORT_DIR


# ## Path setup

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
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
        Moved to `device` and set to eval mode internally.
    dl : DataLoader
        Must be constructed with return_state=True.
    device : str

    Returns
    -------
    Z : (N, latent_dim) float32
    states : (N, 3) float32 -- true (x, y, theta)
    """
    model.to(device).eval()   # move the model too, not just the input batch
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
        theta_flip_pct : percent of frames with heading error > 90 deg, the
            front/back aliasing tail. Chance is 50%; a solved heading is near 0.
            Separates the median (typical accuracy) from the tail (how often the
            encoder gets heading fully backwards), which MAE alone conflates.
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
        "theta_flip_pct":   (ang > (torch.pi / 2)).float().mean().item() * 100.0,
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
    of a uniformly random angle is pi/2 radians = 90 degrees. The flip tail at
    chance is 50% (a random angle is > 90 deg off half the time).
    """
    mean_xy = states_tr[:, :2].mean(dim=0)
    err_xy  = mean_xy.unsqueeze(0) - states_val[:, :2]
    return {
        "x_rmse":           err_xy[:, 0].pow(2).mean().sqrt().item(),
        "y_rmse":           err_xy[:, 1].pow(2).mean().sqrt().item(),
        "pos_rmse":         err_xy.pow(2).sum(1).mean().sqrt().item(),
        "theta_mae_deg":    90.0,
        "theta_median_deg": 90.0,
        "theta_flip_pct":   50.0,
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
            "theta_flip_pct", "x_r2", "y_r2"]
    head = "probe      " + "".join(f"{c:>17s}" for c in cols)
    print(head)
    print("-" * len(head))
    for name in ("chance", "linear", "mlp"):
        if name not in results:
            continue
        m = results[name]
        print(f"{name:11s}" + "".join(f"{m[c]:>17.4f}" for c in cols))


def save_probe_table(results: dict, path: Union[str, Path]) -> None:
    """Write the probe metrics table to a CSV file (mirrors _print_table).

    Parameters
    ----------
    results : dict
        Output of run_probe. Keys chance, linear, mlp.
    path : str or Path
        Output .csv file. Parent directory is created if missing.
    """
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["pos_rmse", "x_rmse", "y_rmse", "theta_mae_deg", "theta_median_deg",
            "theta_flip_pct", "x_r2", "y_r2"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["probe"] + cols)
        for name in ("chance", "linear", "mlp"):
            if name in results:
                writer.writerow([name] + [f"{results[name][c]:.4f}" for c in cols])
    print(f"wrote {path}")


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
    save_dir: Optional[Union[str, Path]] = None,
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
    save_dir : str or Path, optional
        If given, write the metrics table to save_dir/probe_metrics.csv.
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
    if save_dir is not None:
        save_probe_table(results, Path(save_dir) / "probe_metrics.csv")
    return results


# ## Latent visualization
# 
# PCA the 128-D latents down to 2D and scatter them, one panel colored by each true quantity. Dot positions come from the latents only; color is ground truth. Learned quantity = smooth color gradient.
# 
# Look for:
# - x and y panels: smooth gradients running in different directions. The latent cloud laid out as a map of the arena. Warped, not a clean grid, is fine.
# - theta panel: smooth cyclic color = heading encoded. Aliasing shows up as color jumps, or the same hue in two separate regions (two headings the encoder confused).

# In[ ]:


def pca_2d(Z: torch.Tensor) -> tuple:
    """Project latents onto their top 2 principal components.

    PCA is linear: each output coordinate is a fixed weighted sum of the input
    dims. The 2 directions chosen are those of greatest variance in the cloud,
    so this is the same kind of structure a linear probe reads.

    Uses torch's SVD rather than numpy's. numpy-MKL and torch each bundle an
    OpenMP runtime; calling numpy LAPACK (np.linalg.svd) alongside torch can
    load both at once and crash the process. Staying in torch avoids that.

    Parameters
    ----------
    Z : torch.Tensor, shape (N, D)
        Batch of latent vectors.

    Returns
    -------
    coords : np.ndarray, shape (N, 2)
        Each latent projected onto the top 2 components.
    explained_var : float
        Fraction of total variance captured by those 2 components, in [0, 1].
    """
    Zc = Z - Z.mean(dim=0, keepdim=True)            # PCA needs centered data
    _, S, Vh = torch.linalg.svd(Zc, full_matrices=False)
    coords        = (Zc @ Vh[:2].T).cpu().numpy()
    explained_var = float((S[:2] ** 2).sum() / (S ** 2).sum())
    return coords, explained_var


# In[ ]:


def plot_latent_pca(
    Z: torch.Tensor,
    states: torch.Tensor,
    n_points: int = 5000,
    seed: int = 0,
    save_to: Optional[Union[str, Path]] = None,
):
    """Scatter the latent cloud in 2D (PCA), one panel per true quantity.

    Dot positions come only from the latents. Colors come from the ground-truth
    state. A smooth color gradient means the encoder organized that quantity.

    Parameters
    ----------
    Z : torch.Tensor, shape (N, D)
        Latents from extract_latents.
    states : torch.Tensor, shape (N, 3)
        True (x, y, theta) for the same rows as Z.
    n_points : int
        Random subsample size for the scatter. The full set is too dense to read.
    seed : int
        RNG seed for the subsample.
    save_to : str or Path, optional
        Save the figure here if given.

    Returns
    -------
    matplotlib Figure
        The 3-panel figure.
    """
    rng = np.random.default_rng(seed)
    n   = min(n_points, Z.shape[0])
    idx = rng.choice(Z.shape[0], size=n, replace=False)

    coords, evr = pca_2d(Z[idx])
    s = states[idx].cpu().numpy()
    x, y, theta = s[:, 0], s[:, 1], s[:, 2]

    # twilight is cyclic, so theta has no false seam at +/-pi.
    panels = [("x", x, "viridis"), ("y", y, "viridis"), ("theta", theta, "twilight")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, color, cmap) in zip(axes, panels):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=color, cmap=cmap, s=5, alpha=0.5)
        ax.set_title(f"PC space colored by {name}")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        fig.colorbar(sc, ax=ax)

    fig.suptitle(f"latent PCA  (top 2 PCs capture {evr:.1%} of variance)", y=1.02)
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=110, bbox_inches="tight")
        print(f"wrote {save_to}")
    return fig


# ## Latent map (probe axes)
# 
# PCA missed position because SIGReg flattens variance in every direction. Fix: fit a linear probe and look at the latent through the axes it decodes.
# 
# - position panels: predicted (x, y), the arena rebuilt from the latent, colored by true x then true y. Clean gradient aligned with the axis = position decoded.
# - heading panel: predicted angle vs true angle. Tight diagonal = heading decoded. A faint second band offset by ~180 = aliasing (encoder confusing opposite headings). Vertical spread = noise. Colored by angular error.

# In[ ]:


def plot_latent_probe_axes(
    Z: torch.Tensor,
    states: torch.Tensor,
    epochs: int = 40,
    n_points: int = 5000,
    seed: int = 0,
    save_to: Optional[Union[str, Path]] = None,
):
    """View the latent through the axes a linear probe uses to decode state.

    PCA shows the highest-variance directions, but SIGReg flattens variance
    across all dims, so PCA misses position even when a probe decodes it well.
    Here we fit a linear probe and look along the directions it reads:

    - position panels: predicted (x, y) -- the arena rebuilt from the latent.
    - heading panel: predicted angle vs true angle -- a tight diagonal means
      heading is decoded; a band offset by ~180 means the encoder confused
      opposite headings (aliasing); vertical spread is noise.

    Parameters
    ----------
    Z : torch.Tensor, shape (N, D)
        Latents from extract_latents.
    states : torch.Tensor, shape (N, 3)
        True (x, y, theta) for the same rows as Z.
    epochs : int
        Training epochs for the linear probe used to find the axes.
    n_points : int
        Random subsample size for the scatter.
    seed : int
        RNG seed for the probe init and the subsample.
    save_to : str or Path, optional
        Save the figure here if given.

    Returns
    -------
    matplotlib Figure
        The 3-panel figure.
    """
    # fit a linear probe; its outputs are the position/heading readout axes
    torch.manual_seed(seed)
    probe = make_linear_probe(Z.shape[1])
    probe = train_probe(probe, Z, state_to_target(states), epochs=epochs, device="cpu")
    with torch.no_grad():
        pred = probe(Z.to("cpu")).numpy()    # (N, 4): x, y, cos, sin
    s = states.cpu().numpy()

    rng = np.random.default_rng(seed)
    n   = min(n_points, Z.shape[0])
    idx = rng.choice(Z.shape[0], size=n, replace=False)
    pred, s = pred[idx], s[idx]
    pred_x, pred_y, pred_cos, pred_sin = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    true_x, true_y, true_theta = s[:, 0], s[:, 1], s[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    sc0 = axes[0].scatter(pred_x, pred_y, c=true_x, cmap="viridis", s=5, alpha=0.5)
    axes[0].set_title("predicted position, colored by true x")
    axes[0].set_xlabel("predicted x")
    axes[0].set_ylabel("predicted y")
    fig.colorbar(sc0, ax=axes[0])

    sc1 = axes[1].scatter(pred_x, pred_y, c=true_y, cmap="viridis", s=5, alpha=0.5)
    axes[1].set_title("predicted position, colored by true y")
    axes[1].set_xlabel("predicted x")
    axes[1].set_ylabel("predicted y")
    fig.colorbar(sc1, ax=axes[1])

    # heading: predicted angle vs true angle, colored by wrapped angular error.
    # tight diagonal = decoded; band offset by ~180 = aliasing; spread = noise.
    pred_theta = np.arctan2(pred_sin, pred_cos)
    d          = pred_theta - true_theta
    ang_err    = np.degrees(np.abs(np.arctan2(np.sin(d), np.cos(d))))
    sc2 = axes[2].scatter(np.degrees(true_theta), np.degrees(pred_theta),
                          c=ang_err, cmap="viridis_r", s=5, alpha=0.5)
    axes[2].plot([-180, 180], [-180, 180], "k--", lw=1, alpha=0.5)
    axes[2].set_title("predicted vs true heading")
    axes[2].set_xlabel("true theta (deg)")
    axes[2].set_ylabel("predicted theta (deg)")
    fig.colorbar(sc2, ax=axes[2], label="angular error (deg)")

    fig.suptitle("latent viewed along probe-decode axes", y=1.02)
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=110, bbox_inches="tight")
        print(f"wrote {save_to}")
    return fig


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


# ## Visualize the trained latent
# 
# Load the checkpoint, extract val latents, plot. Same SEED as training, so these are the held-out episodes.

# In[ ]:


if __name__ == "__main__":
    if CKPT_PATH.exists():
        train_dl, val_dl = make_dataloaders(
            DATA_PATH, batch_size=256, seed=SEED, return_state=True,
        )
        model = JEPA(grid_size=train_dl.dataset.grid_size, latent_dim=128)
        model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
        Z_val, S_val = extract_latents(model, val_dl, "cpu")

        # archive table + figures for this run under results/<RUN>/
        run_probe(ckpt_path=CKPT_PATH, seed=SEED, save_dir=REPORT_DIR)
        plot_latent_pca(Z_val, S_val, save_to=REPORT_DIR / "latent_pca.png")
        plot_latent_probe_axes(Z_val, S_val, save_to=REPORT_DIR / "latent_probe_axes.png")
        plt.show()
    else:
        print(f"no checkpoint at {CKPT_PATH} - train one first (pipeline.ipynb)")

