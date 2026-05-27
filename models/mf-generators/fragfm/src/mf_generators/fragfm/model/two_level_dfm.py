"""Two-level Discrete Flow Matching for fragment-based generation."""
import torch
import torch.nn as nn


class TwoLevelDFM(nn.Module):
    def __init__(self, vocab_size=10000, hidden_dim=256):
        super().__init__()
        self.fragment_encoder = nn.Embedding(vocab_size, hidden_dim)
        self.molecule_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=4,
        )
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, fragment_ids, molecule_ids):
        frag_emb = self.fragment_encoder(fragment_ids)
        output = self.molecule_decoder(molecule_ids, frag_emb)
        return self.output_proj(output)
