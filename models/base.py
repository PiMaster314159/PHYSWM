"""Shared world-model interface + the pose-supervision loss family."""
import torch, torch.nn as nn, torch.nn.functional as F
from models.components import standardize, unstandardize, state_to_target

class WorldModel(nn.Module):
    def __init__(self, pose_dim: int = 4):
        super().__init__()
        self.pose_dim = pose_dim                          # 4 = [x,y,cos,sin] (unicycle); 5 adds v (bicycle)
        self.register_buffer("pose_mean", torch.zeros(pose_dim))
        self.register_buffer("pose_std",  torch.ones(pose_dim))

    # --- hooks each subclass implements ---
    def encode(self, frame):            raise NotImplementedError
    def predict(self, z, action):       raise NotImplementedError
    def decode_pose(self, z):           raise NotImplementedError
    def representation_loss(self, out, batch, weights):  raise NotImplementedError
    def extra_forward(self, frame, action, next_frame, out):  return {}

    # --- shared machinery ---
    def forward(self, frame, action, next_frame) -> dict:
        z             = self.encode(frame)
        pred_next_z   = self.predict(z, action)
        target_next_z = self.encode(next_frame)
        out = {"z": z, "pred_next_z": pred_next_z, "target_next_z": target_next_z}
        out.update(self.extra_forward(frame, action, next_frame, out))   # recon / phys_next_z / ...
        return out
    
    def pose_supervision(self, out, s_target, s_next_target, w, w_pred):
        m, s = self.pose_mean, self.pose_std
        parts = {}; L = out["z"].new_zeros(())
        if w > 0 and s_target is not None:                       # anchor / readout
            a = F.mse_loss(standardize(self.decode_pose(out["z"]), m, s),
                           standardize(s_target, m, s))
            L = L + w * a; parts["anchor"] = a.item()
        if w_pred > 0 and s_next_target is not None:             # anchor_pred / readout_pred
            ap = F.mse_loss(standardize(self.decode_pose(out["pred_next_z"]), m, s),
                            standardize(s_next_target, m, s))
            L = L + w_pred * ap; parts["anchor_pred"] = ap.item()
        return L, parts

    def loss(self, out, batch, weights) -> tuple:
        rep, parts  = self.representation_loss(out, batch, weights)          # model-specific
        pose, pparts = self.pose_supervision(out, batch.get("s_target"),
                                            batch.get("s_next_target"),
                                            weights.get("anchor", 0.0),
                                            weights.get("anchor_pred", 0.0))   # shared
        total = rep + pose
        parts.update(pparts); parts["total"] = total.item()
        return total, parts

    def rollout_loss(self, batch, weights) -> tuple:
        """Model-agnostic K-step rollout training. Encode frame_0 ONCE, roll predict() over the K held
        actions, and supervise decode_pose at EVERY horizon (standardized, so ego/grounded/jepa all work).
        The first transition also carries the model's full single-step loss (representation grounding +
        1-step pose) via next_frame, so the encoder stays grounded / uncollapsed. Because a wrong dynamics
        parameter COMPOUNDS over the roll, the optimizer gets multi-step gradient the single-step loss
        can't give -- the term meant to sharpen the predictor (and, we hope, rescue JEPA).

        batch: frame (B,1,H,W) · next_frame (B,1,H,W) · actions (B,K,2) · poses (B,K+1,3)
        """
        frame0, next_frame0 = batch["frame"], batch["next_frame"]
        actions, poses = batch["actions"], batch["poses"]
        K = actions.shape[1]
        out = self.forward(frame0, actions[:, 0], next_frame0)                    # first transition
        b0 = {"frame": frame0, "next_frame": next_frame0,
              "s_target": state_to_target(poses[:, 0]), "s_next_target": state_to_target(poses[:, 1])}
        total, parts = self.loss(out, b0, weights)                               # grounding + 1-step pose
        m, s = self.pose_mean, self.pose_std
        z = out["pred_next_z"]; roll = z.new_zeros(())
        for k in range(1, K):                                                    # keep rolling from the prediction
            z = self.predict(z, actions[:, k])
            roll = roll + F.mse_loss(standardize(self.decode_pose(z), m, s),
                                     standardize(state_to_target(poses[:, k + 1]), m, s))
        if K > 1:
            roll = roll / (K - 1)
            total = total + weights.get("rollout", 1.0) * roll
            parts["rollout"] = roll.item(); parts["total"] = total.item()
        return total, parts

    def set_pose_stats(self, mean, std):                    # JEPA calls this at train time
        self.pose_mean.copy_(mean); self.pose_std.copy_(std)