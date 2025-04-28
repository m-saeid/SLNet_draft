import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from encoder.encoder_util import *
except:
    from encoder_util import *

                ##########################################################
                ###################### ResBlocks #########################
                ##########################################################


class Block2(nn.Module):
    def __init__(self, channels, block1_mode, blocks=1,
                 adaptive_ratio=0.9, adaptive_res_dim_ratio=0.25,
                 agmentive_res_dim_ratio=4, bias=True, use_xyz=True):

        super().__init__()
        operation = []
        for _ in range(blocks):
            # operation.append(Residual(channels, res_mode=block1_mode ,res_dim_ratio=res_dim_ratio, bias=bias))
            if block1_mode == 'cross_residual_expert':
                operation.append(CrossResidualExpert(channels, channels, adaptive_ratio=adaptive_ratio,
                                                bottleneck_ratio=adaptive_res_dim_ratio,
                                                augment_expand_ratio=agmentive_res_dim_ratio,
                                                bias=bias))
            elif block1_mode == 'cross_residual':
                operation.append(CrossResidual(channels, channels, adaptive_ratio=adaptive_ratio,
                                                adaptive_res_dim_ratio=adaptive_res_dim_ratio,
                                                agmentive_res_dim_ratio=agmentive_res_dim_ratio, bias=bias))
            elif block1_mode=='adaptive_residual':
                raise ValueError(f"!!!")
                # Residual(ch=out_ch, res_mode=block1_mode, res_dim_ratio=res_dim_ratio, bias=bias)
            elif block1_mode=='residual':
                raise ValueError(f"!!!")
                # Residual(ch=out_ch, res_mode=block1_mode, res_dim_ratio=res_dim_ratio, bias=bias)
            elif block1_mode=='mlp':
                raise ValueError(f"!!!")
                # mlp(in_ch, out_ch)
            else:
                raise ValueError(f"!!!")
        self.operation = nn.Sequential(*operation)

    def forward(self, x):           # (2,32,512)  [b, d, s]
        return self.operation(x)    # (2,32,512)  [b, d, s]  RESIDUAL BLOCKS