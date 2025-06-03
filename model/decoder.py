# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: decoder.py
#   Author  : Qing Wu
#   Email   : wuqing@shanghaitech.edu.cn
#   Date    : 2021/9/20
# -----------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .gridencoder import GridEncoder

# -------------------------------
# decoder implemented by a simple MLP
# -------------------------------
class MLP(nn.Module):
    def __init__(self, in_features=128 + 3, out_features=1, hidden_layers=4, hidden_features=256):
        super(MLP, self).__init__()
        stage_one = []
        stage_two = []
        for i in range(hidden_layers):
            if i == 0:
                stage_one.append(nn.Linear(in_features, hidden_features))
                stage_two.append(nn.Linear(in_features, hidden_features))
            elif i == hidden_layers - 1:
                stage_one.append(nn.Linear(hidden_features, in_features))
                stage_two.append(nn.Linear(hidden_features, out_features))
            else:
                stage_one.append(nn.Linear(hidden_features, hidden_features))
                stage_two.append(nn.Linear(hidden_features, hidden_features))
            stage_one.append(nn.ReLU())
            stage_two.append(nn.ReLU())
        self.stage_one = nn.Sequential(*stage_one)
        self.stage_two = nn.Sequential(*stage_two)

    def forward(self, x):
        h = self.stage_one(x)
        return self.stage_two(x + h)

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        out = torch.sin(self.omega_0 * self.linear(input))
        return out

class Siren(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers, hidden_features,
                 outermost_linear=True, first_omega_0=30, hidden_omega_0=30.0):
                
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        # output = torch.clamp(output, min = -1.0,max = 1.0)
        return output


class Siren_res(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers, hidden_features,
                 outermost_linear=True, first_omega_0=30, hidden_omega_0=30.0):
                
        super().__init__()

        self.net = []
        self.net2 = []

        for i in range(hidden_layers):
            if i==0:
                self.net.append(SineLayer(in_features, hidden_features, 
                                        is_first=True, omega_0=first_omega_0))
                self.net2.append(SineLayer(in_features, hidden_features, 
                                        is_first=True, omega_0=first_omega_0))
            elif i==hidden_layers-1:
                self.net.append(SineLayer(hidden_features, in_features,
                                        is_first=False, omega_0=hidden_omega_0))
                self.net2.append(SineLayer(hidden_features, hidden_features,
                                        is_first=False, omega_0=hidden_omega_0))
            else:
                self.net.append(SineLayer(hidden_features, hidden_features,
                                          is_first=False, omega_0=hidden_omega_0))
                self.net2.append(SineLayer(hidden_features, hidden_features,
                                          is_first=False, omega_0=hidden_omega_0))


        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net2.append(final_linear)
        else:
            self.net2.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
        self.net2 = nn.Sequential(*self.net2)

    def forward(self, coords):
        h = self.net(coords)
        output = self.net2(coords + h)

        # output = torch.clamp(output, min = -1.0,max = 1.0)
        return output

class INGPNetworkRHINO(nn.Module):
    def __init__(self,
                 # For backbone
                 encoding="hashgrid",    # 坐标编码类型
                 num_layers=5,
                 hidden_dim=256,
                 hidden_dim_last=256,
                 #  For encoder
                 input_dim=3,   # 128 + 3
                 multires=6,
                 degree=4,
                 num_levels=16,
                 level_dim=2,
                 base_resolution=10,
                 log2_hashmap_size=19,
                 desired_resolution=2048,
                 align_corners=False,
                 # For transformer
                 freq=9,   # 变换器的频率参数
                 transformer_num_layers=1,
                 transformer_hidden_dim=64,
                 # Skips
                 skips=[4, 8]
                 ):
        super().__init__()

        self.encoder = GridEncoder(input_dim=3, num_levels=num_levels, level_dim=level_dim,
                                   base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size,
                                   desired_resolution=desired_resolution, gridtype='hash', align_corners=align_corners)
        self.transformer = MLPTransformerRHINO(input_dim=3, num_layers=transformer_num_layers,
                                               hidden_dim=transformer_hidden_dim, freq=freq)

        self.in_dim = self.encoder.output_dim + self.transformer.output_dim + input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.hidden_dim_last = hidden_dim_last
        self.skips = skips

        backbone = []

        # Add the in and out layer + the hidden layers inbetween.
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                if l in self.skips:
                    in_dim = self.in_dim  # + self.hidden_dim
                else:
                    in_dim = self.hidden_dim

            if l == num_layers - 2:  # 3
                out_dim = self.hidden_dim_last     # 128
            else:
                out_dim = self.hidden_dim     # 128

            # Bias set to true, False might be better.
            if l == num_layers - 1:
                backbone.append(nn.Linear(in_dim, 1, bias=True))
            else:
                backbone.append(nn.Linear(in_dim, out_dim, bias=True))


        self.backbone = nn.ModuleList(backbone)

    def forward(self, feat ,x):
        # Encode/hash x ...   # x: [B, 3] numData/dimData(x,y,z)
        x = x.view(-1,3)   # (N*K,3)
        h = self.encoder(x)   # 哈希表  (120000,32)
        t = self.transformer.forward(x)   # (120000,3) # 坐标编码
        cf = torch.cat((h, t), dim=1)   # (120000,35)

        cf = torch.cat((feat, cf), dim=1) # 广义的坐标信息融合   (120000,35+128=163)

        # Go through all layers and apply h to them
        for l in range(self.num_layers):
            cf = self.backbone[l](cf)
            if l != self.num_layers - 1:

                cf = F.relu(cf, inplace=False)

                if l + 1 in self.skips:
                    cf = torch.cat((cf, h, t), dim=1)

        colors = torch.sigmoid(cf[:, 0])
        # densities = F.relu(cf[:, 1])

        return colors


class MLPTransformerRHINO(nn.Module):
    def __init__(self, input_dim=3, num_layers=1, hidden_dim=64, freq=9):
        super().__init__()
        self.input_dim = input_dim * freq * 2
        self.output_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.freq = freq

        backbone = []

        # Add the in and out layer + the hidden layers inbetween.
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_dim
            else:
                in_dim = self.hidden_dim

            if l == num_layers - 1:
                out_dim = 3
            else:
                out_dim = self.hidden_dim

            # Bias set to true, False might be better.
            backbone.append(nn.Linear(in_dim, out_dim, bias=True))

        self.backbone = nn.ModuleList(backbone)

    def gamma(self, x):

        # x = x.unsqueeze(-1) # TODO: may need to remove.
        # Create a tensor of powers of 2
        scales = 2.0 ** torch.arange(self.freq)
        # Compute sin and cos features
        features = torch.cat(
            [torch.sin(x * np.pi * scale) for scale in scales] + [torch.cos(x * np.pi * scale) for scale in scales],
            dim=-1)
        return features

    def forward(self, x):

        g = self.gamma(x)

        for l in range(self.num_layers):
            g = self.backbone[l](g)
            if l != self.num_layers - 1:
                # If not last layer, apply ReLU (0+)
                g = F.relu(g, inplace=False)
        return g


