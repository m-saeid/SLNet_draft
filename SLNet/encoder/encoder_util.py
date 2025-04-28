import torch
import torch.nn as nn

import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

try:
    from encoder.encodings import *
    from encoder.mem_block import MEMBlock
except:
    from encodings import *
    from mem_block import MEMBlock

import torch
import torch.nn as nn
import math



# ----------------------------
# 🔹 Utility Modules
# ----------------------------

class DropPath(nn.Module):
    """
    Stochastic Depth: randomly drops entire residual paths.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        mask = torch.rand((x.size(0),) + (1,) * (x.ndim - 1), device=x.device) < keep_prob
        return x.div(keep_prob) * mask

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation: channel-wise recalibration via global context.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N]
        w = x.mean(dim=2, keepdim=True)  # non-parametric reduction (global avg)
        w = self.fc(w)
        return x * w

class TransferMLP(nn.Module):
    """
    Simple 1x1 Conv mapping input channels to output channels.
    """
    def __init__(self, in_ch: int, out_ch: int, bias: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, bias=bias),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

# ----------------------------
# 🔹 CrossResidualExpert Block
# ----------------------------

class CrossResidualExpert(nn.Module):
    """
    Cross-Residual Expert Block:
      - Optional channel transfer
      - Adaptive branch: bottleneck residual
      - Augmentive branch: non-parametric reduction + lightweight MLP
      - Optional SE recalibration (se_mode)
      - Optional dynamic gating
      - Stochastic depth regularization
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        adaptive_ratio: float = 0.9,
        bottleneck_ratio: float = 0.5,
        augment_expand_ratio: float = 4.0,
        transfer: bool = False,
        dynamic_gate: bool = False,
        se_mode: bool = False,
        se_reduction: int = 16,
        drop_prob: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert 0 < adaptive_ratio <= 1, "adaptive_ratio must be in (0,1]"
        self.adaptive_ratio = adaptive_ratio
        self.transfer = transfer
        self.dynamic_gate = dynamic_gate
        self.se_mode = se_mode

        # Transfer layer
        if transfer:
            self.trans = TransferMLP(in_ch, out_ch, bias=bias)
            base_ch = out_ch
        else:
            self.trans = nn.Identity()
            base_ch = in_ch

        # channel splits
        self.ch_ad = int(out_ch * adaptive_ratio)
        self.ch_ag = out_ch - self.ch_ad

        # adaptive branch
        h_ad = max(1, int(base_ch * bottleneck_ratio))
        self.ad_conv = nn.Sequential(
            nn.Conv1d(base_ch, h_ad, 1, bias=bias),
            nn.BatchNorm1d(h_ad),
            nn.ReLU(inplace=True),
            nn.Conv1d(h_ad, self.ch_ad, 1, bias=bias),
            nn.BatchNorm1d(self.ch_ad)
        )
        self.ad_skip = (nn.Conv1d(base_ch, self.ch_ad, 1, bias=bias) if base_ch != self.ch_ad else nn.Identity())
        if se_mode:
            self.ad_se = SELayer(self.ch_ad, reduction=se_reduction)
        self.drop = DropPath(drop_prob)
        self.act = nn.ReLU(inplace=True)

        # augmentive branch: non-parametric reduction + MLP
        if self.ch_ag > 0:
            h_ag = max(1, int(self.ch_ag * augment_expand_ratio))
            self.ag_mlp = nn.Sequential(
                nn.Conv1d(self.ch_ag, h_ag, 1, bias=bias),
                nn.BatchNorm1d(h_ag),
                nn.ReLU(inplace=True),
                nn.Conv1d(h_ag, self.ch_ag, 1, bias=bias),
                nn.BatchNorm1d(self.ch_ag)
            )
            self.ag_skip = (nn.Conv1d(self.ch_ag, self.ch_ag, 1, bias=bias) if False else nn.Identity())
            if se_mode:
                self.ag_se = SELayer(self.ch_ag, reduction=se_reduction)

        # dynamic gating
        if dynamic_gate:
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(base_ch, 2, 1, bias=bias),
                nn.Softmax(dim=1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # 1024,67,32
        x = self.trans(x)                   # Identity      # 1024,67,32
        B, C, N = x.shape

        # dynamic weights
        if self.dynamic_gate:
            w = self.gate(x).view(B, 2, 1, 1)
            w_ad, w_ag = w[:,0], w[:,1]
        else:
            w_ad = w_ag = 1.0

        # adaptive path
        ad = self.ad_conv(x)                            # 67>16>57  1024,57,32
        skip_ad = self.ad_skip(x)                       # 67>57     1024,57,32
        ad = self.act(self.drop(ad * w_ad + skip_ad))   #           1024,57,32
        if self.se_mode:
            ad = self.ad_se(ad)

        # augmentive path
        if self.ch_ag == 0:
            return ad
        # non-parametric reduce: max over points for first ch_ag channels of ad
        # reduced = ad[:, :self.ch_ag, :].max(dim=2, keepdim=True)[0]
        # ag = reduced.expand(-1, -1, N)
        ag = F.adaptive_max_pool1d(ad.permute(0,2,1), self.ch_ag).permute(0,2,1)
        ag = self.act(self.drop(self.ag_mlp(ag) * w_ag + ag))
        if self.se_mode:
            ag = self.ag_se(ag)

        return torch.cat([ad, ag], dim=1)


                ##########################################################
                ###################### Functions #########################
                ##########################################################


'''
def sampling(xyz, s, mode):
    if mode == "fps":
        idx = farthest_point_sample(xyz, s).long()
    return idx
'''

def grouping(k, radius, xyz, new_xyz, mode):
    if mode == "knn":
        idx = knn_point(k, xyz, new_xyz)  # (2,512,24) = knn(24, (2,1024,3), (2,512,3))
    elif mode == "ball":
        idx = query_ball_point(radius, k, xyz, new_xyz)
    return idx


def normalize(norm, f_grouped, f_sampled, xyz_sampled, use_xyz):
    b,_,_ = xyz_sampled.shape
    if norm =="center":   # False
        mean = torch.mean(f_grouped, dim=2, keepdim=True)
    if norm =="anchor":   # True
        mean = torch.cat([f_sampled, xyz_sampled],dim=-1) if use_xyz else f_sampled # True (2,512,19)=cat((2,512,16),(2,512,3))
        mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]    (2,512,1,19)
    std = torch.std((f_grouped-mean).reshape(b,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1) # [b,1,1,1]
    f_grouped = (f_grouped-mean)/(std + 1e-5) #  (2,512,24,19) = (2,512,24,19)/(2,1,1,1)
    return f_grouped


def res_pooling(x, b, n, mode="adaptive_max_pool1d"):
    batch_size = b*n
    if mode == "adaptive_max_pool1d":
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # (k) [b*s,d] 1024,32
        x = x.reshape(b, n, -1).permute(0, 2, 1)             #     [b,d,s] 2,32,512
        return x


def final_pooling(x, mode="adaptive_max_pool1d"):
    if mode == "adaptive_max_pool1d":
        return F.adaptive_max_pool1d(x, 1).squeeze(dim=-1) # [b,d,s]>[b,d]   (2,128,64)>(2,128)


                ##########################################################
                ####################### Classes ##########################
                ##########################################################


class AdaptiveEmbedding(nn.Module):
    """
    Adaptive Embedding Function (AdaptiveEmbedding)
    
    This function implements an adaptive, data-driven variant of the standard
    Gaussian (RBF) embedding. It adjusts the kernel width (sigma) based on the 
    global standard deviation of the input and uses an adaptive blending strategy
    to fuse the Gaussian response with a complementary cosine response.
    
    Assumptions and Implementation Details:
    
    1. Adaptive Kernel Width:
       - Compute a global standard deviation from the input (over points) and 
         adjust the effective sigma as: adaptive_sigma = base_sigma * (1 + global_std).
         
    2. Adaptive Blending:
       - Compute a blend weight (between 0 and 1) as: 
           blend = sigmoid((global_std - baseline) * scaling)
         This weight is used to fuse the Gaussian (RBF) embedding with a cosine embedding.
         
    3. Dynamic Normalization:
       - The difference (tmp) is divided by the adaptive sigma to normalize the scale 
         of the kernel function.
    
    4. Complementarity:
       - The Gaussian captures local similarity via an exponential decay,
         whereas the cosine transformation introduces a periodic component.
         Their fusion is intended to yield a richer representation.
    
    5. Parameterlessness:
       - All adaptation is computed on-the-fly from the data, with no learnable parameters.
    
    Args:
      in_dim (int): Input dimension (typically 3 for XYZ coordinates).
      out_dim (int): Desired output dimension.
      sigma (float): Base sigma value (default kernel width).
      baseline (float): A fixed baseline for computing blend weight (default 0.1).
      scaling (float): Scaling factor for the sigmoid to compute blend (default 10.0).
      eps (float): Small constant to prevent division by zero.
    """
    def __init__(self, in_dim, out_dim, sigma=0.4, baseline=0.1, scaling=10.0, eps=1e-6):
        super(AdaptiveEmbedding, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.base_sigma = sigma  # base kernel width
        self.baseline = baseline
        self.scaling = scaling
        self.eps = eps
        
        feat_dim = math.ceil(out_dim / in_dim)
        self.feat_num = feat_dim * in_dim
        self.out_idx = torch.linspace(0, self.feat_num - 1, out_dim).long()
        # Fixed grid of values for embedding (excluding endpoints)
        self.feat_val = torch.linspace(-1.0, 1.0, feat_dim + 2)[1:-1].reshape(1, -1)
        
    def forward(self, xyz):
        """
        Args:
          xyz: Tensor of shape [B, N, in_dim] or [B, S, K, in_dim]
        Returns:
          Tensor of shape [B, ..., out_dim] computed by adaptively fusing a Gaussian 
          and a cosine response.
        """

        if xyz.shape[-1] != 3:
            xyz = xyz.permute(0,2,1)

        if self.out_dim == 0:
            return xyz
        if xyz.dim() == 3:
            # Compute global standard deviation across points (dim=1)
            global_std = torch.mean(torch.std(xyz, dim=1))
        elif xyz.dim() == 4:
            # Reshape to [B, -1, in_dim] and compute standard deviation over points
            global_std = torch.mean(torch.std(xyz.view(xyz.size(0), -1, self.in_dim), dim=1))
        else:
            raise ValueError("Input must be 3D or 4D")
        
        # Adaptive sigma: scale the base sigma by (1 + global_std)
        adaptive_sigma = self.base_sigma * (1 + global_std)
        # Adaptive blend weight via sigmoid; yields a value in (0,1)
        blend = torch.sigmoid((global_std - self.baseline) * self.scaling)
        
        embeds = []
        for i in range(self.in_dim):
            # Compute difference from fixed grid values
            tmp = xyz[..., i:i+1] - self.feat_val.to(xyz.device)
            # Gaussian (RBF) component using adaptive sigma
            rbf = (-0.5 * (tmp / (adaptive_sigma + self.eps))**2).exp()
            # Cosine component using the same adaptive sigma for scaling
            cosine = torch.cos(tmp / (adaptive_sigma + self.eps))
            # Adaptive fusion of the two components:
            combined = blend * rbf + (1 - blend) * cosine
            embeds.append(combined)
        
        # Concatenate all channels and select the desired output dimensions
        position_embed = torch.cat(embeds, dim=-1)
        position_embed = torch.index_select(position_embed, -1, self.out_idx.to(xyz.device))
        return position_embed.permute(0,2,1)



class mlp(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        super(mlp, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, bias=bias),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class CrossResidual(nn.Module):
    def __init__(self, in_ch, out_ch, adaptive_ratio=0.9, adaptive_res_dim_ratio=0.25, augmentive_res_dim_ratio=4, bias=True):
        super().__init__()

        if adaptive_ratio > 1:
            raise ValueError(f"adaptive_ratio({adaptive_ratio}) must <= 1")

        self.adaptive_ratio = adaptive_ratio
        self.act = nn.ReLU(inplace=True)

        # Adaptive Residual
        in_ch_adaptive_res = in_ch
        hidden_dim_adaptive_res = int(in_ch_adaptive_res * adaptive_res_dim_ratio)
        out_ch_adaptive_res = int(adaptive_ratio*out_ch)

        # augmentive Residual
        if adaptive_ratio != 1:
            in_ch_augmentive_res = out_ch - out_ch_adaptive_res
            hidden_dim_augmentive_res = int(in_ch_augmentive_res * augmentive_res_dim_ratio)
            out_ch_augmentive_res = in_ch_augmentive_res
            self.augmentive_dim_in = in_ch_augmentive_res

        ########################################################
        ################## Adaptive Residual ###################
        self.net1_ad = nn.Sequential(
            nn.Conv1d(in_channels=in_ch_adaptive_res, out_channels=hidden_dim_adaptive_res, kernel_size=1, bias=bias),
            nn.BatchNorm1d(hidden_dim_adaptive_res),
            nn.ReLU(inplace=True))
        self.net2_ad = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim_adaptive_res, out_channels=out_ch_adaptive_res, kernel_size=1, bias=bias),
            nn.BatchNorm1d(out_ch_adaptive_res))

        # Handle skip connection when in_ch != out_ch
        if in_ch_adaptive_res != out_ch_adaptive_res:
            self.skip_ad = nn.Sequential(
                nn.Conv1d(in_channels=in_ch, out_channels=out_ch_adaptive_res, kernel_size=1, bias=bias),
                nn.BatchNorm1d(out_ch_adaptive_res)
            )
        else:
            self.skip_ad = nn.Identity()
        ################## Adaptive Residual ###################
        ########################################################

        ################## Augmentive Residual ##################
        ########################################################
        if adaptive_ratio != 1:
            self.net1_ag = nn.Sequential(
                nn.Conv1d(in_channels=in_ch_augmentive_res, out_channels=hidden_dim_augmentive_res, kernel_size=1, groups=1, bias=bias),
                nn.BatchNorm1d(hidden_dim_augmentive_res),
                nn.ReLU(inplace=True))
            self.net2_ag = nn.Sequential(
                nn.Conv1d(in_channels=hidden_dim_augmentive_res, out_channels=out_ch_augmentive_res, kernel_size=1, bias=bias),
                nn.BatchNorm1d(out_ch_augmentive_res))
        ################## Augmentive Residual ##################
        ########################################################

    def forward(self, x):
        # Adaptive Residual
        residual = self.skip_ad(x)
        out = self.net2_ad(self.net1_ad(x))
        out_adaptive_res = self.act(out + residual)
        # Augmentive Residual
        if self.adaptive_ratio != 1:
            in_augmentive = F.adaptive_max_pool1d(out_adaptive_res.permute(0,2,1), self.augmentive_dim_in).permute(0,2,1)
            out = self.net2_ag(self.net1_ag(in_augmentive))
            out_augmentive_res = self.act(out + in_augmentive)
            # CAT
            out = torch.cat([out_adaptive_res, out_augmentive_res], 1)
        else:
            out = out_adaptive_res

        return out


class AdaptedResidual(nn.Module):
    def init(self, in_ch, out_ch, res_mode='mlp', res_dim_ratio=1.0, bias=True):
        super().init()
        self.act = nn.ReLU(inplace=True)

        if res_mode == "mlp":
            hidden_dim = int(in_ch * res_dim_ratio)
            self.net1 = nn.Sequential(
                nn.Conv1d(in_channels=in_ch, out_channels=hidden_dim, kernel_size=1, bias=bias),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            )
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=hidden_dim, out_channels=out_ch, kernel_size=1, bias=bias),
                nn.BatchNorm1d(out_ch)
            )

            # Handle skip connection when in_ch != out_ch
            if in_ch != out_ch:
                self.skip = nn.Sequential(
                    nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, bias=bias),
                    nn.BatchNorm1d(out_ch)
                )
            else:
                self.skip = nn.Identity()
        else:
            raise Exception(f"Unsupported res_mode: {res_mode}")

    def forward(self, x):
        residual = self.skip(x)
        out = self.net2(self.net1(x))
        return self.act(out + residual)

class Residual(nn.Module):
    def __init__(self, ch, res_mode, res_dim_ratio=1.0, bias=True):
        super().__init__()
        self.act = nn.ReLU(inplace=True)

        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=ch, out_channels=int(ch * res_dim_ratio), kernel_size=1, groups=1, bias=bias),
            nn.BatchNorm1d(int(ch * res_dim_ratio)), nn.ReLU(inplace=True))
        self.net2 = nn.Sequential(
            nn.Conv1d(in_channels=int(ch * res_dim_ratio), out_channels=ch, kernel_size=1, bias=bias),
            nn.BatchNorm1d(ch))

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)



def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def remove_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: indices of the points to remove, [B, S]
    Return:
        new_points: points data with specified indices removed, [B, N-S, C]
    """
    B, N, C = points.shape
    device = points.device
    
    # Create a mask for each batch
    mask = torch.ones((B, N), dtype=torch.bool).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(-1, 1)
    mask[batch_indices, idx] = False

    # Apply mask to remove points and reshape
    new_points = points[mask].view(B, -1, C)
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def edgeSampling(xyz, k, s):
    """
    Input:
        s: sampling numbers
        xyz: all points, [B, N, C]
        k = number of neighbors to attention
    Return:
        samples_idx: sampled pointcloud index, [B, npoint]
    """
    number_of_k = int(xyz.shape[1] * k) # int(s * k)
    sqrdists = square_distance(xyz, xyz)    # 2,1024,1024
    sqrdists = -1*((-1*sqrdists).topk(number_of_k)[0])         # 2,1024,number_ok_k
    distances = sqrdists.sum(dim=-1)        # 2,1024
    idx = distances.topk(s, dim=-1)[1]      # 2,512
    return idx


def sampling(xyz, points, details, S):
    """
    Perform sampling based on details.
    
    Args:
        xyz: Input points, tensor of shape [B, N, 3]
        points: Additional points data, tensor of shape [B, N, d]
        details: List of sampling methods and their parameters
        S: Total number of samples to obtain

    Returns:
        new_xyz: Sampled xyz points, tensor of shape [B, S, 3]
        new_points: Sampled points data, tensor of shape [B, S, d]
    """
    N = xyz.shape[1]
    s = [int(d[-1] * S) for d in details]
    if sum(s) != S:
        diff = S - sum(s)
        s[-1] += diff

    new_xyz = []
    new_points = []
    
    for i in range(len(details)):
        if details[i][0] == "edge":
            idx = edgeSampling(xyz, details[i][1], s[i])
        elif details[i][0] == "fps":
            idx = farthest_point_sample(xyz, s[i]).long()
        else:
            raise ValueError(f"Unknown sampling method: {details[i][0]}")

        new_xyz.append(index_points(xyz, idx))
        new_points.append(index_points(points, idx))

        xyz = remove_points(xyz, idx)
        points = remove_points(points, idx)

    new_xyz = torch.cat(new_xyz, dim=1)
    new_points = torch.cat(new_points, dim=1)

    return new_xyz, new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx






##########################################################
def index_points_(points, idx):
    """
    :param points: points.shape == (B, N, C)
    :param idx: idx.shape == (B, N, K)
    :return:indexed_points.shape == (B, N, K, C)
    """
    raw_shape = idx.shape
    idx = idx.reshape(raw_shape[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.shape[-1]))
    return res.view(*raw_shape, -1)


def knn_(a, b, k):
    """
    :param a: a.shape == (B, N, C)
    :param b: b.shape == (B, M, C)
    :param k: int
    """
    inner = -2 * torch.matmul(a, b.transpose(2, 1))  # inner.shape == (B, N, M)
    aa = torch.sum(a**2, dim=2, keepdim=True)  # aa.shape == (B, N, 1)
    bb = torch.sum(b**2, dim=2, keepdim=True)  # bb.shape == (B, M, 1)
    pairwise_distance = -aa - inner - bb.transpose(2, 1)  # pairwise_distance.shape == (B, N, M)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # idx.shape == (B, N, K)
    return idx


def select_neighbors_(pcd, K, neighbor_type):
    pcd = pcd.permute(0, 2, 1)  # pcd.shape == (B, N, C)
    idx = knn_(pcd, pcd, K)  # idx.shape == (B, N, K)
    neighbors = index_points_(pcd, idx)  # neighbors.shape == (B, N, K, C)
    if neighbor_type == 'neighbor':
        neighbors = neighbors.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    elif neighbor_type == 'diff':
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        neighbors = diff.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    else:
        raise ValueError(f'neighbor_type should be "neighbor" or "diff", but got {neighbor_type}')
    return neighbors


def group(pcd, K, group_type):
    if group_type == 'neighbor':
        neighbors = select_neighbors_(pcd, K, 'neighbor')  # neighbors.shape == (B, C, N, K)
        output = neighbors  # output.shape == (B, C, N, K)
    elif group_type == 'diff':
        diff = select_neighbors_(pcd, K, 'diff')  # diff.shape == (B, C, N, K)
        output = diff  # output.shape == (B, C, N, K)
    elif group_type == 'center_neighbor':
        neighbors = select_neighbors_(pcd, K, 'neighbor')   # neighbors.shape == (B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), neighbors], dim=1)  # output.shape == (B, 2C, N, K)
    elif group_type == 'center_diff':
        diff = select_neighbors_(pcd, K, 'diff')  # diff.shape == (B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), diff], dim=1)  # output.shape == (B, 2C, N, K)
    else:
        raise ValueError(f'group_type should be neighbor, diff, center_neighbor or center_diff, but got {group_type}')
    return output.contiguous()

######################################################s






















"""
# PosE for Raw-point Embedding 
class TPE_old(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, xyz): # B,C,N
        B, _, N = xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)
        
        feat_range = torch.arange(feat_dim).float().cuda() if torch.cuda.is_available() else torch.arange(feat_dim).float()    
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=4).flatten(3)
        position_embed = position_embed.permute(0, 1, 3, 2).reshape(B, self.out_dim, N)
        
        return position_embed
    

class GPE_old(nn.Module):
    def __init__(self, in_dim, out_dim, sigma):
        super(GPE_old, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma

        feat_dim = math.ceil(out_dim / in_dim)
        self.feat_num = feat_dim * self.in_dim
        self.out_idx = torch.linspace(0, self.feat_num - 1, out_dim).long()

        self.feat_val = torch.linspace(-1.0, 1.0, feat_dim + 2)[1:-1].reshape(1, -1)

    def forward(self, xyz):     # 128,1024,3
        # xyz = [B, N, 3] or [B, S, K, 3]

        if xyz.shape[1] == 3:
            xyz = xyz.permute(0,2,1)

        if self.out_dim == 0:
            return xyz

        if xyz.dim() not in {3, 4}:     #3
            raise ValueError("Input must be either [B, in_dim, N] or [B, in_dim, S, K]")

        embeds = []
        # Compute the RBF features for each channel in a loop  /// feat_val: [-0.3333, 0.3333] dayere / sigma: 0.3
        for i in range(self.in_dim):        # 3
            tmp = xyz[..., i : i + 1] - self.feat_val.to(xyz.device)    # [128,1024,2] = [128,1024,1] - [1,2]
            embed = -0.5 * tmp**2 / (self.sigma**2)                     # [128,1024,2] = [128,1024,2] / 0.3
            embeds.append(embed.exp())                                  # [128,1024,2]

        # Concatenate along the last dimension to get all features together
        position_embed = torch.cat(embeds, dim=-1)  # [B, ..., feat_num]    # [128,1024,6]

        # Select the required output dimensions using out_idx
        position_embed = torch.index_select(
            position_embed, -1, self.out_idx.to(xyz.device)
        )
        # [B, ..., out_dim]     [128,1024,6]

        # # Reshape based on the original input dimensions
        # if xyz.dim() == 3:
        #     b, _, n = xyz.shape
        #     position_embed = position_embed.permute(0, 2, 1).reshape(b, self.out_dim, n)
        #     # [B, out_dim, N]
        # elif xyz.dim() == 4:
        #     b, _, s, k = xyz.shape
        #     position_embed = position_embed.permute(0, 3, 1, 2).reshape(
        #         b, self.out_dim, s, k
        #     )
        #     # [B, feat_num, S, K]

        return position_embed.permute(0,2,1)  # [B, ..., out_dim]  [128,1024,6]
"""
