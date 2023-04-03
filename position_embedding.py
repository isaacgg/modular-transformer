import torch
from torch import nn

from interfaces import PositionEmbeddingView


class PositionEmbedding(nn.Module, PositionEmbeddingView):
    def __init__(self, max_len: int, features_len: int) -> None:
        super().__init__()

        self.embedding = nn.Embedding(max_len, features_len)

    def forward(self, x):
        N, seq_len = x.shape

        positions = torch.arange(seq_len).expand(N, seq_len).to(self.device)
        return self.embedding(positions)
