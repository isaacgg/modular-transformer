from typing import List, Optional

import torch
from torch import nn

from interfaces import TransformerBlockView, EmbeddingView, PositionEmbeddingView
from transformer_block import TransformerBlock
from self_attention import SelfAttention


class Encoder(nn.Module):
    def __init__(self, 
                 input_embedding: EmbeddingView,
                 position_embedding: Optional[PositionEmbeddingView],
                 transformers_block: List[TransformerBlockView],
                 dropout: int) -> None:
        super().__init__()

        self.input_embedding = input_embedding
        self.position_embedding = position_embedding

        self.layers = nn.ModuleList(transformers_block)
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x, mask):
        x_embedding = self.input_embedding(x)
        if self.position_embedding:
            x_embedding = x_embedding + self.position_embedding(x_embedding)

        out = self.dropout(x_embedding)        

        # apply transformers_block
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    features_len = 128
    n_transformers = 6

    df = pd.read_csv("E:/Users/isaac/Documents/Apps/crypto_trading/experiments/neural-turing-machines/candle.csv")
    df = df[df["asset_id"] == "USDC-GBP"]
    df.drop(["id", "asset_id"], axis=1, inplace=True)
    df = df.sort_values("time")
    df.pop("time")
    x = df.values
    x = x[None, :]
    x = np.vstack([x, x])
    x = torch.from_numpy(x).float()

    transformers = [TransformerBlock(dropout=0.3,
                                    forward_expansion=256,
                                    self_attention=SelfAttention(features_len=features_len,
                                                                 attention_len=64)) for _ in range(n_transformers)]

    encoder = Encoder(n_layers=3, 
                      dropout=0.3,
                      max_length=len(df),
                      features_len=features_len,
                      transformers_block=transformers)
    nn.Embedding()
    encoder(x)
