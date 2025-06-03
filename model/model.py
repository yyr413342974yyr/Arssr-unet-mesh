# -*- coding:utf-8 -*-
# -----------------------------------------
#   Filename: models.py
#   Author  : Qing Wu
#   Email   : wuqing@shanghaitech.edu.cn
#   Date    : 2021/9/19
# -----------------------------------------
import torch.nn as nn
import torch
import torch.nn.functional as F
from model import encoder
from model import decoder
import numpy as np
import utils

# -------------------------------
# ArSSR model
# -------------------------------
class ArSSR(nn.Module):
    def __init__(self, encoder_name, decoder_name, feature_dim, decoder_depth, decoder_width, freq, embedding=True):
        super(ArSSR, self).__init__()
        self.freq = freq
        self.embedding = embedding

        if encoder_name == 'RDN':
            self.encoder = encoder.RDN(feature_dim=feature_dim)
        if encoder_name == 'SRResnet':
            self.encoder = encoder.SRResnet(feature_dim=feature_dim)
        if encoder_name == 'ResCNN':
            self.encoder = encoder.ResCNN(feature_dim=feature_dim)

        if embedding == 'True':
            self.input_feature_dim = input_feature_dim = 3 * self.freq * 2
        else:
            self.input_feature_dim = input_feature_dim = 3

        if decoder_name == 'MLP':
            self.decoder = decoder.MLP(in_features=feature_dim + input_feature_dim, out_features = 1, hidden_layers=decoder_depth, hidden_features=decoder_width)
        if decoder_name == 'Siren':
            self.decoder = decoder.Siren(in_features=feature_dim + input_feature_dim, out_features = 1, hidden_layers=decoder_depth, hidden_features=decoder_width)
        if decoder_name == 'Siren_res':
            self.decoder = decoder.Siren_res(in_features=feature_dim + input_feature_dim, out_features = 1, hidden_layers=decoder_depth, hidden_features=decoder_width)

    def gamma(self, x):
        # x = x.unsqueeze(-1) # TODO: may need to remove.
        # Create a tensor of powers of 2
        scales = 2.0 ** torch.arange(self.freq)
        # Compute sin and cos features
        features = torch.cat(
            [torch.sin(x * np.pi * scale) for scale in scales] + [torch.cos(x * np.pi * scale) for scale in scales],
            dim=-1)
        return features

    def forward(self, img_lr, xyz_hr):
        """
        :param img_lr: N×1×h×w×d
        :param xyz_hr: N×K×3
        Note that,
            N: batch size  (N in Equ. 3)
            K: coordinate sample size (K in Equ. 3)
            {h,w,d}: dimensional size of LR input image

        """
        feature_map = self.encoder(img_lr)  # N×1×h×w×d
        feature_vector = F.grid_sample(feature_map, xyz_hr.flip(-1).unsqueeze(1).unsqueeze(1),
                                       mode='bilinear',
                                       align_corners=False)[:, :, 0, 0, :].permute(0, 2, 1)

        N, K = xyz_hr.shape[:2]

        if self.embedding == 'True':
            xyz_hr = self.gamma(xyz_hr)

        feature_vector_and_xyz_hr = torch.cat([feature_vector, xyz_hr], dim=-1)  # N×K×(3+feature_dim)   (15,8000,131)

        intensity_pre = self.decoder(feature_vector_and_xyz_hr.reshape(N * K, -1)).reshape(N, K, -1)

        return intensity_pre



class ArSSR_Hash(nn.Module):
    def __init__(self, encoder_name, decoder_name, feature_dim, decoder_depth, decoder_width, freq, embedding=True):
        super(ArSSR_Hash, self).__init__()
        self.freq = freq
        self.embedding = embedding

        if encoder_name == 'RDN':
            self.encoder = encoder.RDN(feature_dim=feature_dim)
        if encoder_name == 'SRResnet':
            self.encoder = encoder.SRResnet(feature_dim=feature_dim)
        if encoder_name == 'ResCNN':
            self.encoder = encoder.ResCNN(feature_dim=feature_dim)

        if embedding == 'True':
            self.input_feature_dim = input_feature_dim = 3 * self.freq * 2
        else:
            self.input_feature_dim = input_feature_dim = 3

        if decoder_name == 'MLP':
            self.decoder = decoder.MLP(in_features=feature_dim + input_feature_dim + 3, out_features = 1, hidden_layers=decoder_depth, hidden_features=decoder_width)
        if decoder_name == 'Siren':
            self.decoder = decoder.Siren(in_features=feature_dim + input_feature_dim + 3, out_features = 1, hidden_layers=decoder_depth, hidden_features=decoder_width)
        if decoder_name == 'Siren_res':
            self.decoder = decoder.Siren_res(in_features=feature_dim + input_feature_dim + 3, out_features = 1, hidden_layers=decoder_depth, hidden_features=decoder_width)

        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((1, 20*20*20, 3))*2 -1),requires_grad = True)

    def gamma(self, x):
        # x = x.unsqueeze(-1) # TODO: may need to remove.
        # Create a tensor of powers of 2
        scales = 2.0 ** torch.arange(self.freq)
        # Compute sin and cos features
        features = torch.cat(
            [torch.sin(x * np.pi * scale) for scale in scales] + [torch.cos(x * np.pi * scale) for scale in scales],
            dim=-1)
        return features

    def forward(self, img_lr, xyz_hr, H, W, D):
        """
        :param img_lr: N×1×h×w×d
        :param xyz_hr: N×K×3
        Note that,
            N: batch size  (N in Equ. 3)
            K: coordinate sample size (K in Equ. 3)
            {h,w,d}: dimensional size of LR input image

        """
        feature_map = self.encoder(img_lr)  # N×1×h×w×d
        feature_vector = F.grid_sample(feature_map, xyz_hr.flip(-1).unsqueeze(1).unsqueeze(1),
                                       mode='bilinear',
                                       align_corners=False)[:, :, 0, 0, :].permute(0, 2, 1)

        N, K = xyz_hr.shape[:2]

        grid = xyz_hr.flip(-1).unsqueeze(1).unsqueeze(1)  #  (b,1,1,N,3)

        if self.embedding == 'True':
            xyz_hr = self.gamma(xyz_hr)

        hash_hr = nn.functional.grid_sample(self.table.reshape(grid.shape[0],H,W,D,3).permute(0,4,3,2,1),grid,mode = "bilinear",padding_mode = 'zeros',align_corners = True).squeeze().permute(1,0)
        hash_hr = hash_hr.unsqueeze(0)
        
        feature_vector_and_xyz_hr = torch.cat([feature_vector, xyz_hr, hash_hr], dim=-1)  # N×K×(3+feature_dim)   (15,8000,131)

        intensity_pre = self.decoder(feature_vector_and_xyz_hr.reshape(N * K, -1)).reshape(N, K, -1)

        return intensity_pre