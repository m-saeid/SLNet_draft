import torch
import torch.nn as nn
import math
try:
    from encoder.encoder_util import group
except:
    from encoder_util import group
    
###############################################################################
# 1. FourierPositionalEncoding
###############################################################################
class FourierPositionalEncoding(nn.Module):
    """
    Computes Fourier (sinusoidal) features using random frequencies.
    
    For each input channel, this module computes:
        sin(2π * B * x) and cos(2π * B * x),
    where B is a fixed random frequency tensor.
    
    If the raw feature dimension (input_dim * 2 * num_frequencies)
    differs from output_dim, a 1×1 convolution projects the features.
    """
    def __init__(self, input_dim, output_dim, num_frequencies=16, scale=1.0):
        super(FourierPositionalEncoding, self).__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        # Fixed random frequencies: shape [input_dim, num_frequencies]
        self.register_buffer('B', torch.randn(input_dim, num_frequencies) * scale)
        raw_feature_dim = input_dim * 2 * num_frequencies
        if raw_feature_dim != output_dim:
            self.proj = nn.Conv1d(raw_feature_dim, output_dim, kernel_size=1)
        else:
            self.proj = nn.Identity()
        
    def forward(self, x):
        # x: [B, input_dim, N]
        B, in_dim, N = x.shape
        x_expanded = x.unsqueeze(-1)  # [B, input_dim, N, 1]
        B_expanded = self.B.unsqueeze(0).unsqueeze(2)  # [1, input_dim, 1, num_frequencies]
        projection = 2 * math.pi * x_expanded * B_expanded  # [B, input_dim, N, num_frequencies]
        sin_features = torch.sin(projection)
        cos_features = torch.cos(projection)
        features = torch.cat([sin_features, cos_features], dim=-1)  # [B, input_dim, N, 2*num_frequencies]
        features = features.permute(0, 1, 3, 2).reshape(B, -1, N)
        return self.proj(features)

###############################################################################
# 2. LearnablePositionalEmbedding
###############################################################################
class LearnablePositionalEmbedding(nn.Module):
    """
    Learns a positional embedding independent of the input coordinates.
    
    The embedding is a learnable parameter of shape [1, output_dim, N] that is 
    expanded along the batch dimension.
    """
    def __init__(self, num_points, output_dim):
        super(LearnablePositionalEmbedding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, output_dim, num_points))
        
    def forward(self, x):
        # x: [B, input_dim, N] (only batch size is used)
        B = x.shape[0]
        return self.pos_embedding.expand(B, -1, -1)

###############################################################################
# 3. RelativePositionalEncoding
###############################################################################
class RelativePositionalEncoding(nn.Module):
    """
    Encodes positions relative to the centroid of the points.
    
    Computes the offset of each point from the centroid and processes it
    with a small MLP.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(RelativePositionalEncoding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # x: [B, input_dim, N]
        centroid = x.mean(dim=-1, keepdim=True)  # [B, input_dim, 1]
        rel = x - centroid                       # [B, input_dim, N]
        rel_encoded = self.mlp(rel.transpose(1, 2))  # [B, N, output_dim]
        return rel_encoded.transpose(1, 2)           # [B, output_dim, N]

###############################################################################
# 4. HarmonicPositionalEncoding
###############################################################################
class HarmonicPositionalEncoding(nn.Module):
    """
    Uses linearly spaced harmonic frequencies for sin/cos encodings.
    
    A 1×1 convolution projects the resulting features to output_dim if necessary.
    """
    def __init__(self, input_dim, output_dim, num_frequencies=4):
        super(HarmonicPositionalEncoding, self).__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.register_buffer('freqs', torch.linspace(1.0, num_frequencies, num_frequencies))
        raw_feature_dim = input_dim * 2 * num_frequencies
        if raw_feature_dim != output_dim:
            self.proj = nn.Conv1d(raw_feature_dim, output_dim, kernel_size=1)
        else:
            self.proj = nn.Identity()
    
    def forward(self, x):
        # x: [B, input_dim, N]
        B, in_dim, N = x.shape
        x_expanded = x.unsqueeze(-1)  # [B, input_dim, N, 1]
        freqs = self.freqs.view(1, 1, 1, self.num_frequencies)  # [1, 1, 1, num_frequencies]
        projection = x_expanded * freqs  # [B, input_dim, N, num_frequencies]
        sin_features = torch.sin(projection)
        cos_features = torch.cos(projection)
        features = torch.cat([sin_features, cos_features], dim=-1)  # [B, input_dim, N, 2*num_frequencies]
        features = features.permute(0, 1, 3, 2).reshape(B, -1, N)
        return self.proj(features)

###############################################################################
# 5. MLPPositionalEncoding
###############################################################################
class MLPPositionalEncoding(nn.Module):
    """
    Maps raw coordinates to an embedding via a lightweight MLP.
    
    The MLP is applied pointwise to each point.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(MLPPositionalEncoding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        # x: [B, input_dim, N] -> transpose to [B, N, input_dim]
        x = x.transpose(1, 2)
        x = self.mlp(x)       # [B, N, output_dim]
        return x.transpose(1, 2)  # [B, output_dim, N]


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


###############################################################################
# 7. ScaledFourierPositionalEncoding
###############################################################################
class ScaledFourierPositionalEncoding(nn.Module):
    """
    A scaled Fourier‐like positional encoding with parameters α and β.
    
    This variant applies a scaling factor to the input coordinates and uses a 
    Fourier transformation (sin/cos) with frequency scaling. If the raw feature 
    dimension (in_dim * 2 * feat_dim) exceeds output_dim, the result is truncated.
    """
    def __init__(self, input_dim, output_dim, alpha, beta):
        super(ScaledFourierPositionalEncoding, self).__init__()
        self.in_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.beta = beta
        # Determine number of frequency components per channel.
        self.feat_dim = output_dim // (input_dim * 2) if output_dim % (input_dim * 2) == 0 else (output_dim // (input_dim * 2)) + 1
        
    def forward(self, x):
        # x: [B, in_dim, N]
        B, C, N = x.shape
        device = x.device
        feat_range = torch.arange(self.feat_dim, device=device).float()  # [feat_dim]
        dim_embed = torch.pow(self.alpha, feat_range / self.feat_dim).view(1, 1, self.feat_dim)  # [1, 1, feat_dim]
        div_embed = (self.beta * x.unsqueeze(-1)) / dim_embed  # [B, in_dim, N, feat_dim]
        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        pos_embed = torch.cat([sin_embed, cos_embed], dim=-1)  # [B, in_dim, N, 2*feat_dim]
        pos_embed = pos_embed.permute(0, 1, 3, 2).reshape(B, self.in_dim * 2 * self.feat_dim, N)
        if pos_embed.shape[1] > self.output_dim:
            pos_embed = pos_embed[:, :self.output_dim, :]
        return pos_embed

###############################################################################
# 8. LinearCoordinateEmbedding (Additional)
###############################################################################
class LinearCoordinateEmbedding(nn.Module):
    """
    Directly projects raw coordinates to a higher-dimensional embedding using a linear layer.
    
    This can serve as a simple baseline.
    """
    def __init__(self, input_dim, output_dim):
        super(LinearCoordinateEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        # x: [B, input_dim, N] -> transpose to [B, N, input_dim]
        x = x.transpose(1, 2)
        x = self.linear(x)
        return x.transpose(1, 2)

    
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



###############################################################################
# Example Usage
###############################################################################
if False: #__name__ == '__main__':
    batch_size = 8
    input_dim = 3       # e.g., xyz coordinates
    num_points = 1024
    output_dim = 64
    sigma = 0.3
    alpha = 100.0
    beta = 1.0

    x = torch.randn(batch_size, input_dim, num_points)

    # Initialize all encoding methods:
    fourier_pos_enc   = FourierPositionalEncoding(input_dim, output_dim, num_frequencies=16, scale=1.0)
    learnable_pos_emb = LearnablePositionalEmbedding(num_points, output_dim)
    relative_pos_enc  = RelativePositionalEncoding(input_dim, output_dim, hidden_dim=64)
    harmonic_pos_enc  = HarmonicPositionalEncoding(input_dim, output_dim, num_frequencies=4)
    mlp_pos_enc       = MLPPositionalEncoding(input_dim, output_dim, hidden_dim=64)
    gaussian_pos_enc  = GaussianPositionalEncoding(input_dim, output_dim, sigma)
    scaled_fourier_enc= ScaledFourierPositionalEncoding(input_dim, output_dim, alpha, beta)
    linear_coord_emb  = LinearCoordinateEmbedding(input_dim, output_dim)
    embedding_cosine = CosineEmbedding(input_dim, output_dim)
    grouped_conv_emb  = GroupedConvolutionalEmbedding(input_dim, output_dim)

    # Forward passes:
    pos_fourier   = fourier_pos_enc(x)
    pos_learnable = learnable_pos_emb(x)
    pos_relative  = relative_pos_enc(x)
    pos_harmonic  = harmonic_pos_enc(x)
    pos_mlp       = mlp_pos_enc(x)
    pos_gaussian  = gaussian_pos_enc(x)
    pos_scaled    = scaled_fourier_enc(x)
    pos_linear    = linear_coord_emb(x)
    pos_cosine    = embedding_cosine(x)
    pos_grouped_conv = grouped_conv_emb(x)

    print("FourierPositionalEncoding shape:     ", pos_fourier.shape)
    print("LearnablePositionalEmbedding shape:    ", pos_learnable.shape)
    print("RelativePositionalEncoding shape:      ", pos_relative.shape)
    print("HarmonicPositionalEncoding shape:      ", pos_harmonic.shape)
    print("MLPPositionalEncoding shape:           ", pos_mlp.shape)
    print("GaussianPositionalEncoding shape:      ", pos_gaussian.shape)
    print("ScaledFourierPositionalEncoding shape: ", pos_scaled.shape)
    print("LinearCoordinateEmbedding shape:       ", pos_linear.shape)
    print("EmbeddingCosine shape:                 ", pos_cosine.shape)
    print("GroupedConvolutionalEmbedding shape:   ", pos_grouped_conv.shape)
