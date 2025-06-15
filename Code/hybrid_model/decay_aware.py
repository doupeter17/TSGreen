import torch
import torch.nn as nn
import torch.nn.functional as F

class DecayAware(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, device="cpu"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.input_embed = nn.Linear(input_size, input_size)
        self.decay_mlp_x = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.Sigmoid()
        )
        self.confidence_layer = nn.Sequential(
            nn.Linear(input_size * 2, input_size),
            nn.Sigmoid()
        )
        self.z_gate = nn.Linear(input_size * 3 + hidden_size, hidden_size)
        self.r_gate = nn.Linear(input_size * 3 + hidden_size, hidden_size)
        self.h_tilde = nn.Linear(input_size * 3 + hidden_size, hidden_size)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size) if output_size else nn.Identity()
        )

    def forward(self, x, x_mask, x_delta, x_mean=None):
        B, T, D = x.shape
        x = torch.nan_to_num(self.input_embed(x), nan=0.0, posinf=0.0, neginf=0.0)
        if x_mean is None:
            x_mean = torch.mean(x, dim=1, keepdim=True).detach()
        x_mean = torch.nan_to_num(x_mean, nan=0.0, posinf=0.0, neginf=0.0)
        h = torch.zeros(B, self.hidden_size, device=self.device)
        outputs = []
        for t in range(T):
            x_t = x[:, t, :]
            m_t = x_mask[:, t, :]
            d_t = torch.clamp(torch.nan_to_num(x_delta[:, t, :], nan=0.0, posinf=100.0, neginf=0.0), 0.0, 100.0)
            gamma_x = self.decay_mlp_x(d_t)
            x_hat = m_t * x_t + (1 - m_t) * (gamma_x * x_mean.squeeze(1))
            confidence = self.confidence_layer(torch.cat([x_hat, d_t], dim=1))
            noise = torch.randn_like(x_hat)
            x_final = torch.nan_to_num(x_hat * confidence + noise * (1 - confidence), nan=0.0, posinf=0.0, neginf=0.0)
            inputs = torch.cat([x_final, m_t, d_t, h], dim=1)
            inputs = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)
            z = torch.sigmoid(self.z_gate(inputs))
            r = torch.sigmoid(self.r_gate(inputs))
            h_tilde = torch.tanh(self.h_tilde(torch.cat([x_final, m_t, d_t, r * h], dim=1)))
            h = (1 - z) * h + z * h_tilde
            h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
            outputs.append(h.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return self.output(torch.nan_to_num(outputs[:, -1, :], nan=0.0, posinf=0.0, neginf=0.0))
