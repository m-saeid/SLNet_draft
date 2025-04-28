import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from encoder.encoder_util import *
except:
    from encoder_util import *

class Agg(nn.Module):
    def __init__(self, mode, **kwargs):
        super(Agg, self).__init__()
        self.mode = mode

    def forward(self, x, b, n):
        batch_size = b*n
        if self.mode == "adaptive_max_pool1d":
            x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # (k) [b*s,d] 1024,32
            x = x.reshape(b, n, -1).permute(0, 2, 1)             #     [b,d,s] 2,32,512
            return x