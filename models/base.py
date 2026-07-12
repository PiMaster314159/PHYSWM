"""Shared world-model interface + the pose-supervision loss family."""
import torch, torch.nn as nn, torch.nn.functional as F
from models.components import standardize, unstandardize

class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("pose_mean", torch.zeros(4))
        self.register_buffer("pose_std",  torch.ones(4))

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

    def set_pose_stats(self, mean, std):                    # JEPA calls this at train time
        self.pose_mean.copy_(mean); self.pose_std.copy_(std)