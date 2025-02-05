# correlation_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TripleLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripleLoss, self).__init__()
        self.margin = margin

    def forward(self, F_anchor, F_pos, F_neg):
        """
        F_anchor, F_pos, F_neg: [B, T, D]
        First, perform averaging over the time/feature dimensions, then compute cosine similarity.
        Use 1 - cosine similarity as the distance.
        """
        # Average pooling to [B, D]
        F_anchor_mean = F_anchor.mean(dim=1)
        F_pos_mean = F_pos.mean(dim=1)
        F_neg_mean = F_neg.mean(dim=1)

        # Calculate distances
        # The higher the cosine similarity, the lower the distance. Here, we define the distance as 1 - cos_sim
        dist_pos = 1 - F.cosine_similarity(F_anchor_mean, F_pos_mean, dim=-1)
        dist_neg = 1 - F.cosine_similarity(F_anchor_mean, F_neg_mean, dim=-1)

        # TL = max(0, dist_pos - dist_neg + margin)
        loss = torch.clamp(dist_pos - dist_neg + self.margin, min=0.0)
        return loss.mean()  # Take the average over the batch
