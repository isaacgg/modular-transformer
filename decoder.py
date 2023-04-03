from typing import Optional, List

import torch
from torch import nn

from interfaces import DecoderView, EmbeddingView, PositionEmbeddingView, DecoderBlockView

class Decoder(nn.Module, DecoderView):
    def __init__(self, 
                 input_embedding: EmbeddingView, 
                 position_embedding: Optional[PositionEmbeddingView],
                 decoder_blocks: List[DecoderBlockView],
                 features_len: int,
                 dropout: int) -> None:
        super().__init__()
        
        self.input_embedding = input_embedding
        self.position_embedding = position_embedding

        self.layers = nn.ModuleList(decoder_blocks)
        self.fc_out = nn.Linear(features_len, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out