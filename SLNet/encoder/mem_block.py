import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class MEMBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, num_groups=4, hidden_ratio=1.0,
                 rnn_type='gru', use_spatial_descriptor=False, use_residual=True):
        """
        MEMBlock: Memory-Enhanced Module inspired by LION (grouped RNNs + projection).

        Args:
            in_channels (int): Input channel dimension.
            out_channels (int or None): Output channel dimension. Defaults to in_channels if None.
            num_groups (int): Number of channel groups for RNN processing.
            hidden_ratio (float): Ratio for internal RNN hidden size relative to group size.
            rnn_type (str): 'gru' or 'lstm'. GRU is preferred for speed.
            use_spatial_descriptor (bool): Whether to apply a 3D Conv to enhance spatial context.
            use_residual (bool): Adds a residual connection if input and output dimensions match.
        """
        super(MEMBlock, self).__init__()
        assert in_channels % num_groups == 0, "in_channels must be divisible by num_groups"

        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.num_groups = num_groups
        self.group_size = in_channels // num_groups
        self.hidden_size = int(self.group_size * hidden_ratio)
        self.use_residual = use_residual and (self.in_channels == self.out_channels)
        self.use_spatial_descriptor = use_spatial_descriptor

        # Optional spatial descriptor (if using 3D input to enhance spatial cues)
        if self.use_spatial_descriptor:
            self.spatial_descriptor = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True)
            )

        # Choose RNN class based on mode ('gru' or 'lstm')
        rnn_cls = nn.GRU if rnn_type.lower() == 'gru' else nn.LSTM

        # Create a list of RNN modules, one per group.
        # Each RNN processes input sequences of dimension [group_size] and outputs hidden states of dimension [hidden_size].
        self.rnn_groups = nn.ModuleList([
            rnn_cls(input_size=self.group_size, hidden_size=self.hidden_size, batch_first=True)
            for _ in range(num_groups)
        ])

        # The output from all groups will have dimension num_groups * hidden_size.
        # We project it to out_channels using a 1x1 conv over the channel dimension.
        self.proj = nn.Conv1d(num_groups * self.hidden_size, self.out_channels, kernel_size=1)
        self.norm = nn.LayerNorm(self.out_channels)

        # Activation after projection and norm.
        self.act = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor. Accepts:
                - [B, C, D, H, W] for volumetric (voxel) features, or
                - [B, C, N] for already flattened features.
        Returns:
            Tensor: Processed tensor of shape [B, out_channels, N] (with N = D*H*W if volumetric).
        """
        B, C = x.size(0), x.size(1)

        # If input is volumetric, optionally apply spatial descriptor and then flatten.
        if x.dim() == 5:
            if self.use_spatial_descriptor:
                x = self.spatial_descriptor(x)  # [B, C, D, H, W]
            x = x.view(B, C, -1)  # [B, C, N] where N = D * H * W
        elif x.dim() != 3:
            raise ValueError("Input tensor must be of shape [B, C, D, H, W] or [B, C, N]")

        # Save original input for residual if enabled.
        residual = x if self.use_residual else None

        # Permute to [B, N, C] for sequential processing via RNNs.
        x = x.permute(0, 2, 1)  # [B, N, C]
        # Split the channel dimension into num_groups groups.
        group_outputs = []
        for i in range(self.num_groups):
            start = i * self.group_size
            end = (i + 1) * self.group_size
            # Extract group i: shape [B, N, group_size]
            x_group = x[:, :, start:end]
            # Process through the i-th RNN
            out, _ = self.rnn_groups[i](x_group)  # out: [B, N, hidden_size]
            group_outputs.append(out)

        # Concatenate outputs from all groups along the channel dimension: [B, N, num_groups * hidden_size]
        x_cat = torch.cat(group_outputs, dim=-1)
        # Permute back to [B, num_groups * hidden_size, N] to match Conv1d input requirements.
        x_cat = x_cat.permute(0, 2, 1)

        # Project the concatenated output to the desired out_channels.
        x_proj = self.proj(x_cat)  # [B, out_channels, N]
        # Apply LayerNorm: the norm expects the last dimension to be the channel dimension,
        # so transpose, norm, and transpose back.
        x_proj = self.norm(x_proj.transpose(1, 2)).transpose(1, 2)
        x_proj = self.act(x_proj)

        # Optionally add a residual connection if shapes match.
        if residual is not None and x_proj.shape == residual.shape:
            x_proj = x_proj + residual

        return x_proj





if __name__ == '__main__':
    B, C, D, H, W = 2, 128, 4, 8, 8
    x = torch.randn(B, C, D, H, W)
    mem = MEMBlock(in_channels=128, out_channels=128, num_groups=4,
                   hidden_ratio=1.0, rnn_type='gru', use_spatial_descriptor=True)
    out = mem(x)
    print("Output shape:", out.shape)  # [B, 128, D*H*W]
