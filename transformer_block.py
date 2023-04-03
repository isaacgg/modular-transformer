import torch
from torch import nn

from interfaces import AttentionView, TransformerBlockView


class TransformerBlock(nn.Module, TransformerBlockView):
    def __init__(self, self_attention: AttentionView, dropout: int, forward_expansion: int) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.features_len = self_attention.get_features_len()
        self.norm_layer1 = nn.LayerNorm(self.features_len)
        self.norm_layer2 = nn.LayerNorm(self.features_len)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.features_len, forward_expansion * self.features_len),
            nn.ReLU(),
            nn.Linear(forward_expansion * self.features_len, self.features_len)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, key_input: torch.Tensor, query_input: torch.Tensor, value_input: torch.Tensor, mask: torch.Tensor=None):
        attention = self.self_attention(key_input, query_input, value_input, mask)
        x = self.dropout(self.norm_layer1(attention + query_input))
        
        forward = self.feed_forward(x)
        return self.dropout(self.norm_layer2(forward + x))
