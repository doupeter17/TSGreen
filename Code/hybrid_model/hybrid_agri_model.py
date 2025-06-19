import torch
import torch.nn as nn
from .decay_aware import DecayAware
from .time_embedding_gru import TimeEmbeddingGRUEncoder
from .tabular_encoder import TabularEncoder
from .cross_attention import CrossModelAttentionFusion

class HybridAgriModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.soil_encoder = DecayAware(config['soil_in'], config['soil_hidden'], config['branch_out'], device=config['device'])
        self.env_encoder = TimeEmbeddingGRUEncoder(config['indoor_in'] + config['weather_in'], 32, config['branch_out'])
        self.crop_encoder = TabularEncoder(config['crop_in'], 16, config['branch_out'])
        self.fusion = CrossModelAttentionFusion([config['branch_out']] * 3, config['fusion_dim'])
        self.head = nn.Sequential(
            nn.LayerNorm(config['branch_out'] * 3),
            nn.Dropout(0.3),
            nn.Linear(config['branch_out'] * 3, 64),
            nn.ReLU(),
            nn.Linear(64, config['output_dim']),
            nn.Softplus()
        )

    def forward(self, soil, mask, delta, indoor, weather, crop):
        f1 = self.soil_encoder(soil, mask, delta)
        f2 = self.env_encoder(torch.cat([indoor, weather], dim=2), delta[:, :, 0])
        f3 = self.crop_encoder(crop)
        return self.head(self.fusion([f1, f2, f3]))
