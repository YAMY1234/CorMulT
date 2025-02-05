import math

import torch
import torch.nn as nn

def make_positions(tensor, padding_idx):
    """Generate position numbers for non-padding symbols in the input tensor.
    Position numbers begin at padding_idx + 1.
    """
    mask = tensor.ne(padding_idx).int()
    positions = (torch.cumsum(mask, dim=1) + padding_idx) * mask
    return positions.long()

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, padding_idx=0, init_size=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = dict()   # device --> actual weight; due to nn.DataParallel :-(
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float32).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        device = input.device
        if device not in self.weights or max_pos > self.weights[device].size(0):
            self.weights[device] = self.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            ).to(device)
        self.weights[device] = self.weights[device].type_as(self._float_tensor)
        positions = make_positions(input, self.padding_idx)
        # Ensure positions are within the range of embeddings
        positions = positions.clamp(max=self.weights[device].size(0) - 1)
        position_embeddings = self.weights[device].index_select(0, positions.view(-1))
        return position_embeddings.view(bsz, seq_len, -1).detach()