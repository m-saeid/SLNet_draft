import torch
import torch.nn as nn
import math
try:
    from encoder.encoder_util import group
except:
    from encoder_util import group

###############################################################################
# 5. GaussianPositionalEncoding
###############################################################################
class GaussianPositionalEncoding(nn.Module):
    """
    Computes Gaussian (RBF) features for each coordinate channel.
    
    For each channel, a set of fixed RBF centers (sampled from -1 to 1) is used.
    The raw RBF features are computed as:
        exp(-0.5 * ((x - c) / sigma)^2)
    A subset of these features is selected to produce an output of shape [B, output_dim, N].
    """
    def __init__(self, input_dim, output_dim, sigma):
        super(GaussianPositionalEncoding, self).__init__()
        self.in_dim = input_dim
        self.output_dim = output_dim
        self.sigma = sigma
        
        self.feat_dim = math.ceil(output_dim / input_dim)
        self.feat_num = self.feat_dim * input_dim
        out_idx = torch.linspace(0, self.feat_num - 1, output_dim).long()
        self.register_buffer('out_idx', out_idx)
        feat_val = torch.linspace(-1.0, 1.0, self.feat_dim + 2)[1:-1]
        self.register_buffer('feat_val', feat_val)
    
    def forward(self, x):
        # Accepts input as [B, in_dim, N] or [B, N, in_dim]; ensure channels are in the second dimension.
        if x.dim() == 3:
            if x.shape[1] != self.in_dim and x.shape[-1] == self.in_dim:
                x = x.permute(0, 2, 1)
        else:
            raise ValueError("GaussianPositionalEncoding expects a 3D tensor [B, in_dim, N] or [B, N, in_dim]")
        x = x.transpose(1, 2)  # [B, N, in_dim]
        B, N, _ = x.shape
        x_expanded = x.unsqueeze(-1)  # [B, N, in_dim, 1]
        feat_val = self.feat_val.view(1, 1, 1, self.feat_dim)
        diff = x_expanded - feat_val  # [B, N, in_dim, feat_dim]
        rbf = torch.exp(-0.5 * (diff ** 2) / (self.sigma ** 2))
        # Use reshape instead of view to handle non-contiguous tensors.
        rbf_flat = rbf.reshape(B, N, self.feat_num)  # [B, N, in_dim * feat_dim]
        rbf_selected = torch.index_select(rbf_flat, dim=-1, index=self.out_idx)
        return rbf_selected.transpose(1, 2)  # [B, output_dim, N]


    
class CosineEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_features = math.ceil(out_dim / in_dim)
        self.out_idx = torch.linspace(0, self.num_features * in_dim - 1, out_dim).long()

    def forward(self, xyz):
        # Transpose to ensure the input is [B, N, in_dim]
        xyz = xyz.transpose(1, 2)  # [B, N, in_dim]
        lin_vals = torch.linspace(-math.pi, math.pi, self.num_features + 2, device=xyz.device)[1:-1].reshape(1, 1, -1)
        features = []
        for i in range(self.in_dim):
            tmp = xyz[..., i:i+1] - lin_vals
            features.append(torch.cos(tmp))
        feature = torch.cat(features, dim=-1)
        feature = torch.index_select(feature, -1, self.out_idx.to(xyz.device))
        # Reshape to [B, N, out_dim] and then transpose to [B, out_dim, N]
        return feature.transpose(1, 2)


class GroupedConvolutionalEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GroupedConvolutionalEmbedding, self).__init__()
        self.K = 32
        self.group_type = 'center_diff'
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim*2, out_dim, 1, bias=False), nn.BatchNorm2d(out_dim), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(out_dim, out_dim//2, 1, bias=False), nn.BatchNorm2d(out_dim//2), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(out_dim, out_dim, 1, bias=False), nn.BatchNorm2d(out_dim), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(out_dim, out_dim//2, 1, bias=False), nn.BatchNorm2d(out_dim//2), nn.LeakyReLU(0.2))

    def forward(self, x):
        x_list = []
        x = group(x, self.K, self.group_type)  # (B, C=3, N) -> (B, C=6, N, K)
        x = self.conv1(x)  # (B, C=6, N, K) -> (B, C=128, N, K)
        x = self.conv2(x)  # (B, C=128, N, K) -> (B, C=64, N, K)
        x = x.max(dim=-1, keepdim=False)[0]  # (B, C=64, N, K) -> (B, C=64, N)
        x_list.append(x)
        x = group(x, self.K, self.group_type)  # (B, C=64, N) -> (B, C=128, N, K)
        x = self.conv3(x)  # (B, C=128, N, K) -> (B, C=128, N, K)
        x = self.conv4(x)  # (B, C=128, N, K) -> (B, C=64, N, K)
        x = x.max(dim=-1, keepdim=False)[0]  # (B, C=64, N, K) -> (B, C=64, N)
        x_list.append(x)
        x = torch.cat(x_list, dim=1)  # (B, C=128, N)
        return x

