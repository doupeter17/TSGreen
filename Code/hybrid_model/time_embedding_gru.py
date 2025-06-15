import torch
import torch.nn as nn

class TimeEmbeddingGRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.time_embed = nn.Linear(1, 8)
        self.gru = nn.GRU(input_dim + 8, hidden_dim, batch_first=True)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, delta_t):
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        delta_t = torch.nan_to_num(delta_t, nan=0.0, posinf=0.0, neginf=0.0)
        time_feature = self.time_embed(delta_t.unsqueeze(-1))
        x_cat = torch.cat([x, time_feature], dim=-1)
        _, h = self.gru(x_cat)
        return self.out(torch.nan_to_num(h[-1], nan=0.0, posinf=0.0, neginf=0.0))
