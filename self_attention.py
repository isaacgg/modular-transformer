import torch
from torch import nn

from interfaces import AttentionView


class SelfAttention(nn.Module, AttentionView):
    def __init__(self, features_len: int, n_heads: int, attention_len: int):
        super().__init__()
        
        self.features_len = features_len
        self.n_heads = n_heads
        self.attention_len = attention_len
        
        self.q_w = nn.Parameter(torch.empty(self.n_heads, self.features_len, self.attention_len))
        self.k_w = nn.Parameter(torch.empty(self.n_heads, self.features_len, self.attention_len))
        self.v_w = nn.Parameter(torch.empty(self.n_heads, self.features_len, self.attention_len))
        
        self.fc_out = nn.Linear(self.attention_len, self.features_len)

        self._init_variables()
    
    def get_features_len(self) -> int:
        return self.features_len

    def _init_variables(self):
        torch.nn.init.xavier_uniform(self.q_w)
        torch.nn.init.xavier_uniform(self.k_w)
        torch.nn.init.xavier_uniform(self.v_w)

    def _calculate_attention_vectors(self, x, weights):
        return torch.einsum('bsf,hfa->bsha', x, weights)  # batch matrix multiplication
    
    def forward(self, key_input: torch.Tensor, query_input: torch.Tensor, value_input: torch.Tensor, mask: torch.Tensor=None):
        keys = self._calculate_attention_vectors(key_input, self.k_w)  # [batch_size, sequence_len, heads, attention_len]
        queries = self._calculate_attention_vectors(query_input, self.q_w) 
        values = self._calculate_attention_vectors(value_input, self.v_w)

        scores = torch.softmax(torch.einsum("bqha,bkha->bhqk", queries, keys) / np.sqrt(self.attention_len), dim=2)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        return torch.einsum("bhqs,bsha->bsha", scores, values)


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    df = pd.read_csv("E:/Users/isaac/Documents/Apps/crypto_trading/experiments/neural-turing-machines/candle.csv")
    df = df[df["asset_id"] == "USDC-GBP"]
    df.drop(["id", "asset_id"], axis=1, inplace=True)
    df = df.sort_values("time")
    df.pop("time")
    x = df.values
    x = x[None, :]
    x = np.vstack([x, x])
    x = torch.from_numpy(x).float()
    model = SelfAttention(features_len=5, n_heads=3, attention_len=8)
    model(x, x, x)