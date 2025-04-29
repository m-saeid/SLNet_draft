import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder.encoder_util import mlp, square_distance, index_points
from encoder.m6_block2 import Block2

from encoder.encodings import *


class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, out_channel, de_fp_fuse='mlp', de_fp_block='mlp', blocks=1, res_expansion=1.0, bias=True):
        super(FeaturePropagation, self).__init__()

        self.fuse = mlp(in_channel, out_channel, bias=bias)

        sigma = 0.3
        alpha = 100.0
        beta = 1.0

        input_dim, output_dim = in_channel, out_channel

        if de_fp_fuse == 'mlp':
            self.embd = mlp(input_dim, output_dim, bias=bias)
        elif de_fp_fuse == 'adaptive':
            self.embd = AdaptiveEmbedding(input_dim, output_dim, sigma, alpha, beta)
        elif de_fp_block == 'hybrid':
            self.embd = HybridEmbedding(input_dim, output_dim, sigma, alpha, beta)
        elif de_fp_fuse == 'fourier':
            self.embd = FourierPositionalEncoding(input_dim, output_dim, num_frequencies=16, scale=1.0)
        elif de_fp_fuse == 'scaled_fourier':
            self.embd = ScaledFourierPositionalEncoding(input_dim, output_dim, alpha, beta)
        elif de_fp_fuse == 'gaussian':
            self.embd = GaussianPositionalEncoding(input_dim, output_dim, sigma)  
        elif de_fp_fuse == 'harmonic':
            self.embd = HarmonicPositionalEncoding(input_dim, output_dim, num_frequencies=4)
        elif de_fp_fuse == 'mlp2':
            self.embd = MLPPositionalEncoding(input_dim, output_dim, hidden_dim=64)
        elif de_fp_fuse == 'learnable':
            self.embd = LearnablePositionalEmbedding(num_points, output_dim) 
        elif de_fp_fuse == 'relative':
            self.embd = RelativePositionalEncoding(input_dim, output_dim, hidden_dim=64)
        elif de_fp_fuse == 'linear_coord':
            self.embd = LinearCoordinateEmbedding(input_dim, output_dim)
        else:
            raise Exception(f"de_fp_fuse!!! {de_fp_fuse}")   


        self.extraction = Block2(out_channel, de_fp_block, blocks, res_dim_ratio=res_expansion, bias=bias)

    def forward(self, xyz1, xyz2, points1, points2):
        # B N 3   -  B S 3  -  B D' N    -  B D'' S
        # 2 32 3  -  2 8 3  -  2 128 32  -  2 128 8
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D''', N]
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1) # B S D''   2 8 128 
        B, N, C = xyz1.shape               # B N 3     2 32 3
        _, S, _ = xyz2.shape               # B S 3      2 8 3

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)  #  2 8 128 > 
        else:
            dists = square_distance(xyz1, xyz2)            # (B N 3, B S 3) > B N S  -  2 32 8
            dists, idx = dists.sort(dim=-1)                # B N S , B N S  -  2 32 8 , 2 32 8
            dists, idx = dists[:, :, :3], idx[:, :, :3]    # k=3  B N 3 , B N 3  -  2 32 3 - 2 32 3

            dist_recip = 1.0 / (dists + 1e-8)                   # B N 3  -  2 32 3
            norm = torch.sum(dist_recip, dim=2, keepdim=True)   # B N 1  -  2 32 1
            weight = dist_recip / norm                          # B N 3  -  2 32 3
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2) # B N D''  2 32 128

        if points1 is not None:                     # True
            points1 = points1.permute(0, 2, 1)      # [B N D']   2 32 128
            new_points = torch.cat([points1, interpolated_points], dim=-1)  # [B N D"""]     2 32 256
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)    # B D""" N    2 256 32

        new_points = self.fuse(new_points)          #   B D* N > B D** N
        # MLP1(256>512; [2,256,32]>[2,512,32])     MLP2(576>256; [2,576,128]>[2,256,128])
        # MLP3(288>128; [2,288,512]>[2,128,512])   MLP4(144>128; [2,144,2048]>[2,128,2048])
        new_points = self.extraction(new_points)
        # 2 512 32 > 2 512 32  -  2 256 128 > 2 256 128  -  2 128 512 > 2 128 512  -  2 128 2048 > 2 128 2048
        return new_points
        # 2 512 32