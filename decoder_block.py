import torch
from torch import nn

from interfaces import AttentionView, TransformerBlockView, DecoderBlockView


class DecoderBlock(nn.Module, DecoderBlockView):
    def __init__(self, attention_layer: AttentionView, transformer_block: TransformerBlockView, dropout:int) -> None:
        super().__init__()

        features_len = transformer_block.features_len

        self.norm = nn.LayerNorm(features_len)
        self.attention = attention_layer
        self.transformer_block = transformer_block
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


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
