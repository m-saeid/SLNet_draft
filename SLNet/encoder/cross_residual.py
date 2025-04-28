import torch
import torch.nn as nn
import torch.nn.functional as F

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
        se_mode: bool = True,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.trans(x)
        B, C, N = x.shape

        # dynamic weights
        if self.dynamic_gate:
            w = self.gate(x).view(B, 2, 1, 1)
            w_ad, w_ag = w[:,0], w[:,1]
        else:
            w_ad = w_ag = 1.0

        # adaptive path
        ad = self.ad_conv(x)
        skip_ad = self.ad_skip(x)
        ad = self.act(self.drop(ad * w_ad + skip_ad))
        if self.se_mode:
            ad = self.ad_se(ad)

        # augmentive path
        if self.ch_ag == 0:
            return ad
        # non-parametric reduce: max over points for first ch_ag channels of ad
        reduced = ad[:, :self.ch_ag, :].max(dim=2, keepdim=True)[0]
        ag = reduced.expand(-1, -1, N)
        ag = self.act(self.drop(self.ag_mlp(ag) * w_ag + ag))
        if self.se_mode:
            ag = self.ag_se(ag)

        return torch.cat([ad, ag], dim=1)

# ----------------------------
# 🔹 Test Script
# ----------------------------

def test_cre():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, Cin, N, Cout = 2, 32, 128, 64
    for cfg in [
        {'adaptive_ratio':0.9,'transfer':False,'dynamic_gate':False,'se_mode':True},
        {'adaptive_ratio':1.0,'transfer':True,'dynamic_gate':True,'se_mode':False},
        {'adaptive_ratio':0.5,'transfer':True,'dynamic_gate':False,'se_mode':True},
    ]:
        print('CFG',cfg)
        m = CrossResidualExpert(Cin, Cout,
            adaptive_ratio=cfg['adaptive_ratio'],
            transfer=cfg['transfer'],
            dynamic_gate=cfg['dynamic_gate'],
            se_mode=cfg['se_mode']
        ).to(device)
        x = torch.randn(B,Cin,N,device=device)
        y = m(x)
        print(f"=> {x.shape} -> {y.shape}")

if __name__=='__main__':
    test_cre()
