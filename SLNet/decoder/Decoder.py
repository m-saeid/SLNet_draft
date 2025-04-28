import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("E:\Papers\SLNet\code\SLNet")
sys.path.append("E:\Papers\SLNet\code\SLNet\decoder")

from encoder.Encoder import Encoder
from encoder.encoder_util import mlp, square_distance, index_points
from encoder.encodings import *

from decoder.m0_fpropagation import FeaturePropagation


class Decoder(nn.Module):
    def __init__(self,
                 
                 task="partseg_shapenet",

                 # Encoder:
                 n=1024,
                 embed=[6,16,'mlp'],   # mlp, fourier, scaled_fourier, gaussian, harmonic, mlp2, relative, linear_coord, /learnable
                 res_dim_ratio=1.0,
                 bias=True,
                 use_xyz=True,
                 norm_mode="center",
                 std_mode="BN1D",
                 dim_ratio=[2, 2, 2, 2],

                 num_blocks1=[2, 2, 2, 2],
                 transfer_mode = ['mlp', 'mlp', 'mlp', 'mlp'], # mlp, fourier, scaled_fourier, gaussian, harmonic, mlp2, relative, linear_coord, /learnable
                 block1_mode = ['mlp', 'mlp', 'gaussian', 'mlp'], # mlp, fourier, scaled_fourier, gaussian, harmonic, mlp2, relative, linear_coord, /learnable

                 num_blocks2=[2, 2, 2, 2],
                 block2_mode = ['mlp', 'mlp', 'mlp', 'mlp'], # mlp, fourier, scaled_fourier, gaussian, harmonic, mlp2, relative, linear_coord, /learnable

                 k_neighbors=[32, 32, 32, 32],
                 sampling_mode=['fps', 'fps', 'fps', 'fps'],
                 sampling_ratio=[2, 2, 2, 2],

                 # Decoder:
                 de_dims=[512, 256, 128, 128],
                 de_blocks=[2, 2, 2, 2],

                 de_fp_fuse=['mlp', 'mlp', 'mlp', 'mlp'], # mlp, fourier, scaled_fourier, gaussian, harmonic, mlp2, relative, linear_coord, /learnable
                 de_fp_block=['mlp', 'mlp', 'mlp', 'mlp'], # mlp, fourier, scaled_fourier, gaussian, harmonic, mlp2, relative, linear_coord, /learnable

                 gmp_dim=64,
                 gmp_dim_mode = 'mlp',

                 cls_dim=64,
                 cls_map_mode = 'mlp', # mlp, fourier, scaled_fourier, gaussian, harmonic, mlp2, relative, linear_coord, /learnable
                 gmp_map_end_mode = 'mlp',

                 num_cls = 50,
                 classifier_mode = 'mlp', # mlp, fourier, scaled_fourier, gaussian, harmonic, mlp2, relative, linear_coord, /learnable
                 **kwargs):
        
        super(Decoder, self).__init__()

        self.task = task

        self.encoder = Encoder(
                            n=n,
                            embed=embed,   # mlp, fourier, scaled_fourier, gaussian, harmonic, mlp2, relative, linear_coord, /learnable
                            res_dim_ratio=res_dim_ratio,
                            bias=bias,
                            use_xyz=use_xyz,
                            norm_mode=norm_mode,
                            std_mode=std_mode,
                            dim_ratio=dim_ratio,

                            num_blocks1=num_blocks1,
                            transfer_mode = transfer_mode, # mlp, fourier, scaled_fourier, gaussian, harmonic, mlp2, relative, linear_coord, /learnable
                            block1_mode = block1_mode, # mlp, fourier, scaled_fourier, gaussian, harmonic, mlp2, relative, linear_coord, /learnable

                            num_blocks2=num_blocks2,
                            block2_mode=block2_mode, # mlp, fourier, scaled_fourier, gaussian, harmonic, mlp2, relative, linear_coord, /learnable

                            k_neighbors=k_neighbors,
                            sampling_mode=sampling_mode,
                            sampling_ratio=sampling_ratio,
                            )


        # en_dims = [16,32,64,128,128]
        en_dims = [embed[1]]
        for i in dim_ratio:
            en_dims.append(en_dims[-1]*i)

        ### Building Decoder #####
        self.decode_list = nn.ModuleList()
        en_dims.reverse()
        de_dims.insert(0,en_dims[0])
        assert len(en_dims) == len(de_dims) == len(de_blocks)+1
        for i in range(len(en_dims)-1):
            self.decode_list.append(
                FeaturePropagation(de_dims[i]+en_dims[i+1], de_dims[i+1], de_fp_fuse=de_fp_fuse[i], de_fp_block=de_fp_block[i],
                                           blocks=de_blocks[i], res_expansion=res_dim_ratio,
                                           bias=bias)
            )

        sigma = 0.3
        alpha = 100.0
        beta = 1.0

        # class label mapping
        if task == 'partseg_shapenet':
            if cls_map_mode == "mlp":
                self.cls_map = nn.Sequential(
                    mlp(16, cls_dim, bias=bias),
                    mlp(cls_dim, cls_dim, bias=bias)
                    )
            elif cls_map_mode == 'fourier':
                self.cls_map = nn.Sequential(
                    FourierPositionalEncoding(input_dim=16, output_dim=cls_dim, num_frequencies=16, scale=1.0),
                    FourierPositionalEncoding(input_dim=cls_dim, output_dim=cls_dim, num_frequencies=16, scale=1.0)
                    )
            elif cls_map_mode == 'scaled_fourier':
                self.cls_map = nn.Sequential(
                    ScaledFourierPositionalEncoding(input_dim=16, output_dim=cls_dim, alpha=alpha, beta=beta),
                    ScaledFourierPositionalEncoding(input_dim=cls_dim, output_dim=cls_dim, alpha=alpha, beta=beta)
                    )
            elif cls_map_mode == 'gaussian':
                self.cls_map = nn.Sequential(
                    GaussianPositionalEncoding(input_dim=16, output_dim=cls_dim, sigma=sigma),
                    GaussianPositionalEncoding(input_dim=cls_dim, output_dim=cls_dim, sigma=sigma)
                    )
            elif cls_map_mode == 'harmonic':
                self.cls_map = nn.Sequential(
                    HarmonicPositionalEncoding(input_dim=16, output_dim=cls_dim, num_frequencies=4),
                    HarmonicPositionalEncoding(input_dim=cls_dim, output_dim=cls_dim, num_frequencies=4)
                    )
            elif cls_map_mode == 'mlp2':
                self.cls_map = nn.Sequential(
                    MLPPositionalEncoding(input_dim=16, output_dim=cls_dim, hidden_dim=64),
                    MLPPositionalEncoding(input_dim=cls_dim, output_dim=cls_dim, hidden_dim=64)
                    )
            #elif cls_map_mode == 'learnable':
            #    self.embd = LearnablePositionalEmbedding(num_points, output_dim) 
            #    self.cls_map = nn.Sequential(
            #        LearnablePositionalEmbedding(num_points, output_dim) (16, cls_dim, bias=bias),
            #        LearnablePositionalEmbedding(num_points, output_dim) (cls_dim, cls_dim, bias=bias)
            #        )
            elif cls_map_mode == 'relative':
                self.cls_map = nn.Sequential(
                    RelativePositionalEncoding(input_dim=16, output_dim=cls_dim, hidden_dim=64),
                    RelativePositionalEncoding(input_dim=cls_dim, output_dim=cls_dim, hidden_dim=64)
                    )
            elif cls_map_mode == 'linear_coord':
                self.cls_map = nn.Sequential(
                    LinearCoordinateEmbedding(input_dim=16, output_dim=cls_dim),
                    LinearCoordinateEmbedding(input_dim=cls_dim, output_dim=cls_dim)
                    )
            else:
                raise Exception(f"cls_map_mode!!! {cls_map_mode}")

        elif task == "semseg_s3dis":
            cls_dim = 0

        # global max pooling mapping
        self.gmp_map_list = nn.ModuleList()
        for en_dim in en_dims:
            if gmp_dim_mode == "mlp":
                self.gmp_map_list.append(mlp(en_dim, gmp_dim, bias=bias))

            elif gmp_dim_mode == 'fourier':
                self.gmp_map_list.append(
                    FourierPositionalEncoding(input_dim=en_dim, output_dim=gmp_dim, num_frequencies=16, scale=1.0)
                )
            elif gmp_dim_mode == 'scaled_fourier':
                self.gmp_map_list.append(
                    ScaledFourierPositionalEncoding(input_dim=en_dim, output_dim=gmp_dim, alpha=alpha, beta=beta)
                )
            elif gmp_dim_mode == 'gaussian':
                self.gmp_map_list.append(
                    GaussianPositionalEncoding(input_dim=en_dim, output_dim=gmp_dim, sigma=sigma) 
                )
            elif gmp_dim_mode == 'harmonic':
                self.gmp_map_list.append(
                    HarmonicPositionalEncoding(input_dim=en_dim, output_dim=gmp_dim, num_frequencies=4)
                )
            elif gmp_dim_mode == 'mlp2':
                self.gmp_map_list.append(
                    MLPPositionalEncoding(input_dim=en_dim, output_dim=gmp_dim, hidden_dim=64)
                )
            #elif gmp_dim_mode == 'learnable':
            #    self.gmp_map_list.append(
            #        LearnablePositionalEmbedding(num_points, output_dim) 
            #    )
            elif gmp_dim_mode == 'relative':
                self.gmp_map_list.append(
                    RelativePositionalEncoding(input_dim=en_dim, output_dim=gmp_dim, hidden_dim=64)
                )
            elif gmp_dim_mode == 'linear_coord':
                self.gmp_map_list.append(
                    LinearCoordinateEmbedding(input_dim=en_dim, output_dim=gmp_dim)
                )
            else:
                raise Exception(f"gmp_dim_mode!!! {gmp_dim_mode}")

        if gmp_map_end_mode == "mlp":
            self.gmp_map_end = mlp(gmp_dim*len(en_dims), gmp_dim, bias=bias)
        elif gmp_map_end_mode == 'fourier':
            self.gmp_map_end = FourierPositionalEncoding(input_dim=gmp_dim*len(en_dims), output_dim=gmp_dim, num_frequencies=16, scale=1.0)
        elif gmp_map_end_mode == 'scaled_fourier':
            self.gmp_map_end = ScaledFourierPositionalEncoding(input_dim=gmp_dim*len(en_dims), output_dim=gmp_dim, alpha=alpha, beta=beta)
        elif gmp_map_end_mode == 'gaussian':
            self.gmp_map_end = GaussianPositionalEncoding(input_dim=gmp_dim*len(en_dims), output_dim=gmp_dim, sigma=sigma)  
        elif gmp_map_end_mode == 'harmonic':
            self.gmp_map_end = HarmonicPositionalEncoding(input_dim=gmp_dim*len(en_dims), output_dim=gmp_dim, num_frequencies=4)
        elif gmp_map_end_mode == 'mlp2':
            self.gmp_map_end = MLPPositionalEncoding(input_dim=gmp_dim*len(en_dims), output_dim=gmp_dim, hidden_dim=64)
        #elif gmp_map_end_mode == 'learnable':
        #    self.gmp_map_end = LearnablePositionalEmbedding(num_points, output_dim) 
        elif gmp_map_end_mode == 'relative':
            self.gmp_map_end = RelativePositionalEncoding(input_dim=gmp_dim*len(en_dims), output_dim=gmp_dim, hidden_dim=64)
        elif gmp_map_end_mode == 'linear_coord':
            self.gmp_map_end = LinearCoordinateEmbedding(input_dim=gmp_dim*len(en_dims), output_dim=gmp_dim)
        else:
            raise Exception(f"gmp_map_end_mode!!! {gmp_map_end_mode}")


        # classifier
        if classifier_mode == "mlp":
            self.classifier = nn.Sequential(
                nn.Conv1d(gmp_dim+cls_dim+de_dims[-1], 128, 1, bias=bias),
                nn.BatchNorm1d(128), nn.Dropout(),
                nn.Conv1d(128, num_cls, 1, bias=bias)
            )
        elif classifier_mode == 'fourier':
            self.classifier = nn.Sequential(
                FourierPositionalEncoding(input_dim=gmp_dim+cls_dim+de_dims[-1], output_dim=128, num_frequencies=16, scale=1.0),
                nn.BatchNorm1d(128), nn.Dropout(),
                FourierPositionalEncoding(input_dim=128, output_dim=num_cls, num_frequencies=16, scale=1.0)
            )
        elif classifier_mode == 'scaled_fourier':
            self.classifier = nn.Sequential(
                ScaledFourierPositionalEncoding(input_dim=gmp_dim+cls_dim+de_dims[-1], output_dim=128, alpha=alpha, beta=beta),
                nn.BatchNorm1d(128), nn.Dropout(),
                ScaledFourierPositionalEncoding(input_dim=128, output_dim=num_cls, alpha=alpha, beta=beta)
            )
        elif classifier_mode == 'gaussian':
            self.classifier = nn.Sequential(
                GaussianPositionalEncoding(input_dim=gmp_dim+cls_dim+de_dims[-1], output_dim=128, sigma=sigma),
                nn.BatchNorm1d(128), nn.Dropout(),
                GaussianPositionalEncoding(input_dim=128, output_dim=num_cls, sigma=sigma)
            )
        elif classifier_mode == 'harmonic':
            self.classifier = nn.Sequential(
                HarmonicPositionalEncoding(input_dim=gmp_dim+cls_dim+de_dims[-1], output_dim=128, num_frequencies=4),
                nn.BatchNorm1d(128), nn.Dropout(),
                HarmonicPositionalEncoding(input_dim=128, output_dim=num_cls, num_frequencies=4)
            )
        elif classifier_mode == 'mlp2':
            self.classifier = nn.Sequential(
                MLPPositionalEncoding(input_dim=gmp_dim+cls_dim+de_dims[-1], output_dim=128, hidden_dim=64),
                nn.BatchNorm1d(128), nn.Dropout(),
                MLPPositionalEncoding(input_dim=128, output_dim=num_cls, hidden_dim=64)
            )
        #elif classifier_mode == 'learnable':
        #    self.classifier = nn.Sequential(
        #        LearnablePositionalEmbedding(num_points, output_dim),
        #        nn.BatchNorm1d(128), nn.Dropout(),
        #        LearnablePositionalEmbedding(num_points, output_dim)
        #    )
        elif classifier_mode == 'relative':
            self.classifier = nn.Sequential(
                RelativePositionalEncoding(input_dim=gmp_dim+cls_dim+de_dims[-1], output_dim=128, hidden_dim=64),
                nn.BatchNorm1d(128), nn.Dropout(),
                RelativePositionalEncoding(input_dim=128, output_dim=num_cls, hidden_dim=64)
            )
        elif classifier_mode == 'linear_coord':
            self.classifier = nn.Sequential(
                LinearCoordinateEmbedding(input_dim=gmp_dim+cls_dim+de_dims[-1], output_dim=128),
                nn.BatchNorm1d(128), nn.Dropout(),
                LinearCoordinateEmbedding(input_dim=128, output_dim=num_cls)
            )
        else:
            raise Exception(f"classifier_mode!!! {classifier_mode}")


        self.en_dims = en_dims

    def forward(self, x, norm_plt, cls_label): 
        if self.task == 'partseg_shapenet': # x:[B,C,N]  norm_plt:[B,C,N]  cls_label:[B,num_cls]  dataset=shapenet
            xyz = x                                     # [B, C, N]  (2,3,2048)
            x = torch.cat([x, norm_plt],dim=1)          # [B, 6, N]  (2,6,2048)
        elif self.task == 'semseg_s3dis':  # x: [B,9,N] norm_plt:None  cls_label:None  dataset=s3dis
            xyz = x[:, :3, :]                           # [B, C, N]  (2,3,4096)
            # x                                           [B, 9, N]  (2,9,4096)
        else:
            raise Exception(f"Task!!! {self.task}")

        xyz_list, x_list = self.encoder(xyz, x)
        # xyz_list                                  # [B, S, C]  (2, 2048,1024,512,256,128, 3)
        # x_list                                    # [B, D, S]  (2, 16,32,64,128,128, 2048,1024,512,256,128)

        trans_feat = x_list[-1]

        # Decoder
        xyz_list.reverse()              # len = 5     [B, S, C]  (2, 128,256,512,1024,2048, 3)
        x_list.reverse()                # len = 5     [B, D, S]  (2, 128,128,64,32,16, 128,256,512,1024,2048)
        x = x_list[0]                   #             [B, D, S]  (2 128 8)
        for i in range(len(self.decode_list)): # 4
            x = self.decode_list[i](xyz_list[i+1], xyz_list[i], x_list[i+1],x)        # > [B, D, S]
            #                       2 256 3   , 2 128 3   , 2 128 256  , 2 512 256      > 2 512 256
            #                       2 512 3   , 2 256 3   , 2 64 512   , 2 256 512      > 2 256 512
            #                       2 1024 3  , 2 512 3   , 2 32 1024  , 2 128 1024     > 2 128 1024
            #                       2 2048 3  , 2 1024 3  , 2 16 2048  , 2 128 2048     > 2 128 2048

        # Global Context
        # x_list    : 2 128 128  -  2 128 256  -  2 64 512  -  2 32 1024  -  2 16 2048
        gmp_list = []
        for i in range(len(x_list)):    # 5
            gmp_list.append(F.adaptive_max_pool1d(self.gmp_map_list[i](x_list[i]), 1))
            #print(x_list[i].shape, self.gmp_map_list[i], self.gmp_map_list[i](x_list[i]).shape, F.adaptive_max_pool1d(self.gmp_map_list[i](x_list[i]), 1).shape)
            #print()
            # 1: 2 128 128   MLP(128>64) 2 64 128      pool    2 64 1
            # 2: 2 128 256   MLP(128>64) 2 64 256      pool    2 64 1
            # 3: 2 64 512    MLP(128>64) 2 64 512      pool    2 64 1
            # 4: 2 32 1024   MLP(32>64)  2 64 1024     pool    2 64 1
            # 5: 2 16 2048   MLP(16>64)  2 64 2048     pool    2 64 1
            # gmp_list: 2 64 1, 2 64 1, 2 64 1, 2 64 1, 2 64 1

        global_context = self.gmp_map_end(torch.cat(gmp_list, dim=1)) # 2,320,1  MLP(320>64)  2,64,1

        # cls_token
        if cls_label is not None: #shapenet cls_label:[B,num_cls] (32,16)
            cls_token = self.cls_map(cls_label.unsqueeze(dim=-1))  # [b, cls_dim, 1]  2,16 > 2,16,1  MLP(16>64) MLP(64>64)  2,64,1
            x = torch.cat([x, global_context.repeat([1, 1, x.shape[-1]]), cls_token.repeat([1, 1, x.shape[-1]])], dim=1) # 2 256 2048
            # 2 128 2048  -  2 64 2048  -  2 64 2048  >  2 256 2048
        else:
            x = torch.cat([x, global_context.repeat([1, 1, x.shape[-1]])], dim=1) # 2 256 2048


        x = self.classifier(x)      # 2 256 2048 MLP(256>128>50) 2 50 2048
        x = F.log_softmax(x, dim=1) # 2 50 2048
        x = x.permute(0, 2, 1)      # 2 2048 50
        return x, trans_feat        # 2 4096 13     2 128 128  s3dis


if __name__ == '__main__':

    def all_params(model):
        return sum(p.numel() for p in model.parameters())
    def trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    data = torch.rand(2, 3, 2048) # 2 3 2048
    norm = torch.rand(2, 3, 2048) # 2 3 2048
    cls_label = torch.rand([2, 16]) # 2 16
    print("===> testing modelD ...")
    decoder = Decoder(task="partseg_shapenet")

    print(f'params: {trainable_params(decoder)}')

    out = decoder(data, norm, cls_label)  # [2,2048,50]
    
    print(out[0].shape, out[1].shape)