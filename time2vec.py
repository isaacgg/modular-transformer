import torch
from torch import nn

from interfaces import EmbeddingView


class Time2Vec(nn.Module, EmbeddingView):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.parameters_linear = nn.Linear(in_features=in_features, out_features=1, bias=True)
        self.parameters_periodic = nn.Linear(in_features=in_features, out_features=out_features - 1, bias=True)

        self.function = lambda x: torch.complex(real=torch.cos(x), imag=torch.sin(x))
    
    def forward(self, x):
        return self.parameters_linear(x) + self.function(self.parameters_periodic(x))


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
  
    model = Time2Vec(in_features=5, out_features=30)
    a = model(x)
    print(a)
