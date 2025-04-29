import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from encoder.encoder_util import *
except:
    from encoder_util import *


class Grouping(nn.Module):
    def __init__(self, k, use_xyz=True, **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param s: s number
        :param k: k-nerighbors
        :param kwargs: others
        """
        super(Grouping, self).__init__()
        self.k = k
        self.use_xyz = use_xyz

    def forward(self, xyz, f, xyz_sampled, f_sampled): # 2,1024,3  2,1024,16
        B, N, C = xyz.shape         # 2,1024,3
        xyz = xyz.contiguous()  # 2,1024,3    xyz [btach, n, xyz]

        # GROPPING
        idx = grouping(self.k, 0, xyz, xyz_sampled, mode="knn")  # (2,512,24) = knn(24, (2,1024,3), (2,512,3))
        xyz_grouped = index_points(xyz, idx)        # [b, s, k, c]  (2,512,24,3)
        f_grouped = index_points(f, idx)  # [b, s, k, c]  (2,512,24,16)

        return xyz_grouped, f_grouped