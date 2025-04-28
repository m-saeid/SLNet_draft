import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from encoder.encoder_util import *
    from encoder.encodings import *
    from encoder.mem_block import MEMBlock
except:
    from encoder_util import *
    from encodings import *
    from mem_block import MEMBlock

                ##########################################################
                ##################### RedResBlock ########################
                ##########################################################
'''
    fourier_pos_enc   = FourierPositionalEncoding(input_dim, output_dim, num_frequencies=16, scale=1.0) # fourier
    learnable_pos_emb = LearnablePositionalEmbedding(num_points, output_dim)                            # learnable
    relative_pos_enc  = RelativePositionalEncoding(input_dim, output_dim, hidden_dim=64)                # relative
    harmonic_pos_enc  = HarmonicPositionalEncoding(input_dim, output_dim, num_frequencies=4)            # harmonic
    mlp_pos_enc       = MLPPositionalEncoding(input_dim, output_dim, hidden_dim=64)                     # mlp2
    gaussian_pos_enc  = GaussianPositionalEncoding(input_dim, output_dim, sigma)                        # gaussian
    scaled_fourier_enc= ScaledFourierPositionalEncoding(input_dim, output_dim, alpha, beta)             # scaled_fourier
    linear_coord_emb  = LinearCoordinateEmbedding(input_dim, output_dim)                                # linear_coord
                        mlp(in_ch, out_ch, bias=bias)                                                   # mlp
'''


class Block1(nn.Module):    # reduce d  and  residual
    def __init__(self, ch, out_ch, transfer_mode='mlp', block1_mode='mlp',  blocks=1, res_dim_ratio=1, bias=True, use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param ch:
        :param blocks:
        
        fourier_pos_enc   = FourierPositionalEncoding(input_dim, output_dim, num_frequencies=16, scale=1.0) # fourier
        learnable_pos_emb = LearnablePositionalEmbedding(num_points, output_dim)                            # learnable
        relative_pos_enc  = RelativePositionalEncoding(input_dim, output_dim, hidden_dim=64)                # relative
        harmonic_pos_enc  = HarmonicPositionalEncoding(input_dim, output_dim, num_frequencies=4)            # harmonic
        mlp_pos_enc       = MLPPositionalEncoding(input_dim, output_dim, hidden_dim=64)                     # mlp2
        gaussian_pos_enc  = GaussianPositionalEncoding(input_dim, output_dim, sigma)                        # gaussian
        scaled_fourier_enc= ScaledFourierPositionalEncoding(input_dim, output_dim, alpha, beta)             # scaled_fourier
        linear_coord_emb  = LinearCoordinateEmbedding(input_dim, output_dim)                                # linear_coord
                            #mlp(in_ch, out_ch, bias=bias)                                                  # mlp
        """

        super().__init__()
        in_ch = 3+2*ch if use_xyz else 2*ch

        input_dim, output_dim = in_ch, out_ch

        sigma = 0.3
        alpha = 100.0
        beta = 1.0

        if transfer_mode == 'mlp':
            self.transfer = mlp(in_ch, out_ch, bias=bias)
        elif transfer_mode[:3] == 'mem':
            mode = list(map(lambda x: str(x), transfer_mode.split('_')))
            mode = {'num_groups':int(mode[1]), 'hidden_ratio':float(mode[2]), 'rnn_type':mode[3],
                    'use_spatial_descriptor':True if mode[4]=='True' else False, 'use_residual':True if mode[5]=='True' else False}
            self.embd = MEMBlock(input_dim, output_dim, mode['num_groups'], mode['hidden_ratio'], mode['rnn_type'], mode['use_spatial_descriptor'], mode['use_residual'])
        
        else:
            raise Exception(f"transfer_mode!!! {transfer_mode}")
                    
        operation = []
        for _ in range(blocks):
            operation.append(
                Residual(ch=out_ch, res_mode=block1_mode, res_dim_ratio=res_dim_ratio, bias=bias)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, k, d = x.size()       # [b,s,k,d] 2,512,24,35
        x = x.permute(0, 1, 3, 2)   # [b,s,d,k] 2,512,35,24
        x = x.reshape(-1, d, k)     # [b*s,d,k] 1024,35,24
        x = self.transfer(x)        # [b*s,d,k] 1024,32,24   transfer: Conv1d(35>32)+BN+ReLU
        #batch_size, _, _ = x.size() # [b*s,d,k] 1024,32,24
        x = self.operation(x)       # [b*s,d,k] 1024,32,24   RESIDUAL BLOCKS
        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # (k) [b*s,d] 1024,32
        # x = x.reshape(b, n, -1).permute(0, 2, 1)             #     [b,d,s] 2,32,512
        return x          # 1024,32,24          #[b,d,s] 2,32,512