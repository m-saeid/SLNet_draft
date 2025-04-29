import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from encoder.encoder_util import *
except:
    from encoder_util import *



class Normalization(nn.Module):
    def __init__(self, d, norm_mode="anchor", std_mode="1111", use_xyz=True, alpha_beta="yes_ab", **kwargs): # norm_mode: "anchor" or "center" or "nearest_to_mean" | std_mode: "1111" or "B111" or "BN11" or "BN1D"
        super(Normalization, self).__init__()
        self.use_xyz = use_xyz
        self.std_mode=std_mode
        self.alpha_beta = alpha_beta
        if norm_mode.lower() == 'center':
            self.mode = 'center'
        elif norm_mode.lower() == 'anchor':
            self.mode = 'anchor'
        elif norm_mode.lower() == 'nearest_to_mean':
            self.mode = 'nearest_to_mean'
        else:
            print(f"Unrecognized Normalization.mode: {norm_mode}! >>> None.")
            self.mode = None
        if self.mode is not None:
            add_d=3 if self.use_xyz else 0
        if self.alpha_beta=='yes_ab' or self.alpha_beta=='yes_ba':
            self.affine_alpha = nn.Parameter(torch.ones([1,1,1,d + add_d]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, d + add_d]))
        
    def forward(self, xyz_sampled, f_sampled, xyz_grouped, f_grouped):
        b,s,k,_ = xyz_grouped.shape
        # USE_XYZ
        f_grouped = torch.cat([f_grouped, xyz_grouped],dim=-1) if self.use_xyz else f_grouped
        # NORMALIZE
        if self.mode is not None: # f_grouped [2,N,k,D] (2,512,24,19)
            if self.mode =="center":   # False
                mean = torch.mean(f_grouped, dim=2, keepdim=True)
            elif self.mode =="anchor":   # True
                mean = torch.cat([f_sampled, xyz_sampled],dim=-1) if self.use_xyz else f_sampled # True (2,512,19)=cat((2,512,16),(2,512,3))
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]    (2,512,1,19)
            elif self.mode =="nearest_to_mean":
                # Find nearest point in the nighborhood to mean in the nighborhood
                dist = torch.sum((f_grouped - torch.mean(f_grouped, dim=2, keepdim=True))**2, dim=-1)  # [B,N,k] (2,512,24)
                idx = torch.argmin(dist, dim=2, keepdim=True)                       # [B,N,1] (2,512,1)
                mean = torch.gather(f_grouped, 2, idx.unsqueeze(-1).expand(-1, -1, -1, f_grouped.size(-1)))  # [B,N,1,D] (2,512,1,19)
            else:
                raise ValueError(f"Unrecognized mode: {self.mode}! >>> None.")

            if self.std_mode == "1111":
                std = torch.std((f_grouped-mean).reshape(-1), dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1) # [1,1,1,1]
            elif self.std_mode == "B111":
                std = torch.std((f_grouped-mean).reshape(b,-1), dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1) # [b,1,1,1]
            elif self.std_mode == "BN11":
                std = torch.std((f_grouped-mean).reshape(b, s, -1), dim=-1, keepdim=True).unsqueeze(dim=-1) # [b, s, 1, 1]
            elif self.std_mode == "BN1D":
                std = torch.std((f_grouped-mean).permute(0,1,3,2), dim=-1, keepdim=True).permute(0,1,3,2) # [b, s, 1, d]
            else:
                raise ValueError(f"Unrecognized std_mode: {self.std_mode}! >>> None.")

            f_grouped = (f_grouped-mean)/(std + 1e-5) #  (2,512,24,19) = (2,512,24,19)/(2,1,1,1)
            
        if self.alpha_beta=="yes_ab":
            f_grouped = self.affine_alpha * f_grouped + self.affine_beta
        if self.alpha_beta=="yes_ba":
            f_grouped = self.affine_alpha * (f_grouped + self.affine_beta)

        f_grouped = torch.cat([f_grouped, f_sampled.view(b, s, 1, -1).repeat(1, 1, k, 1)], dim=-1)
        return f_grouped       # 2,512,24,35
    
if __name__ == "__main__":
    # Sample input data
    b, s, k, d = 2, 512, 24, 16  # batch size, number of points, number of neighbors, feature dimension
    xyz_sampled = torch.rand(b, s, 3)
    f_sampled = torch.rand(b, s, d)
    xyz_grouped = torch.rand(b, s, k, 3)
    f_grouped = torch.rand(b, s, k, d)

    # Instantiate the Normalization class
    normalization = Normalization(d=d, norm_mode="nearest_to_mean", std_mode="BN1D", use_xyz=True)# norm_mode: "anchor" or "center" or "nearest_to_mean" | std_mode: "1111" or "B111" or "BN11" or "BN1D"

    # Run the forward pass
    output = normalization(xyz_sampled, f_sampled, xyz_grouped, f_grouped)

    # Print the output shape
    print("Output shape:", output.shape)
    