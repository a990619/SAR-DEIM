# Copyright (c) 2024 The DEIM Authors. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register  # 注意相对路径是两点 ..core

__all__ = ["DEIM"]


# ---------- Common ----------
def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=bias
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else (act if isinstance(act, nn.Module) else nn.Identity())
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ---------- ILDA：两层差分注意力（通道门 + 空间门）----------
class ILDA(nn.Module):
    """
    Inter-Layer Difference Attention for adjacent scales (P4,P5).
    1) 对齐通道到 c_mid；2) 上采样 P5 到 P4 分辨率；3) 计算差分；4) 通道门 × 空间门；5) 残差式回注。
    """

    def __init__(self, c4_in, c5_in, c_mid=256, rd=4):
        super().__init__()
        self.align4 = Conv(c4_in, c_mid, k=1, act=True)
        self.align5 = Conv(c5_in, c_mid, k=1, act=True)

        # channel gate: GAP+GMP -> 1x1-MLP -> sigmoid，输出 [B,C,1,1]
        hidden = max(c_mid // rd, 16)
        self.mlp = nn.Sequential(
            nn.Conv2d(c_mid * 2, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c_mid, 1, bias=True),
            nn.Sigmoid(),
        )

        # spatial gate: depthwise 3x3 -> pointwise 1x1 -> sigmoid，输出 [B,1,H,W]
        self.spatial = nn.Sequential(
            nn.Conv2d(c_mid, c_mid, kernel_size=3, stride=1, padding=1, groups=c_mid, bias=True),
            nn.Conv2d(c_mid, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

        # 融合（p4 加权、p4 残差、p5 上采样） -> c_mid
        self.fuse = Conv(c_mid * 3, c_mid, k=1, act=True)
        self.gamma = nn.Parameter(torch.tensor(0.0))  # 初值 0，训练更稳

    def forward(self, p4, p5):
        p4a = self.align4(p4)                              # [B,C,H,W]
        p5a = self.align5(p5)                              # [B,C,H/2,W/2]
        p5a_up = F.interpolate(p5a, size=p4a.shape[-2:], mode="nearest")

        delta = p4a - p5a_up                               # 差分，保留符号

        gap = F.adaptive_avg_pool2d(delta, 1)
        gmp = F.adaptive_max_pool2d(delta, 1)
        ch_gate = self.mlp(torch.cat([gap, gmp], dim=1))   # [B,C,1,1]
        sp_gate = self.spatial(delta)                      # [B,1,H,W]

        w = ch_gate * sp_gate                              # 广播得到 [B,C,H,W]
        y = self.fuse(torch.cat([p4a * w, p4a, self.gamma * p5a_up], dim=1))
        return y  # [B,C_mid,H,W]


# ---------- ABF2：两节点双向融合（P5->P4 top-down；P4->P5 bottom-up）----------
class ABF2(nn.Module):
    def __init__(self, c_in4, c_in5, c_mid=256):
        super().__init__()
        # 对齐到统一通道
        self.l4 = Conv(c_in4, c_mid, k=1, act=True)
        self.l5 = Conv(c_in5, c_mid, k=1, act=True)

        # Top-Down @P4: 2 路 softmax 权重（来自 GAP 特征）
        self.td_mlp = nn.Sequential(
            nn.Conv2d(c_mid * 2, max(c_mid // 4, 16), 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(c_mid // 4, 16), 2, 1, bias=True),
        )

        # Bottom-Up @P5: 三路凸组合标量权重
        self.bu_w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-4

        # 下采样 3×3,s=2
        self.down = Conv(c_mid, c_mid, k=3, s=2, act=True)

        # 输出再映射（接口保持）
        self.out4 = Conv(c_mid, c_mid, k=1, act=True)
        self.out5 = Conv(c_mid, c_mid, k=1, act=True)

    def forward(self, p4, p5):
        p4 = self.l4(p4)                      # [B,C,H,W]
        p5 = self.l5(p5)                      # [B,C,H/2,W/2]
        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode="nearest")

        # --- Top-Down @P4 ---
        td_logits = self.td_mlp(torch.cat([F.adaptive_avg_pool2d(p4, 1),
                                           F.adaptive_avg_pool2d(p5_up, 1)], dim=1))  # [B,2,1,1]
        td_w = F.softmax(td_logits.flatten(1), dim=1).unsqueeze(-1).unsqueeze(-1)      # [B,2,1,1]
        p4_td = td_w[:, :1] * p4 + td_w[:, 1:] * p5_up

        # --- Bottom-Up @P5 ---
        w = F.relu(self.bu_w)
        w = w / (w.sum() + self.eps)
        p4_td_down = self.down(p4_td)
        p4_down = self.down(p4)
        p5_bu = w[0] * p5 + w[1] * p4_td_down + w[2] * p4_down

        return self.out4(p4_td), self.out5(p5_bu)


# ---------- WSDC：共享核多空洞上下文 ----------
class WSDC(nn.Module):
    """
    Weighted Shared-Dilation Convolution (三分支 dilation=1/3/5，共享同一套 3x3 卷积核)。
    - 先把输入降到 c_red（默认 c/2），做三条不同 dilation 的分支，但共享同一组卷积权重。
    - 每条分支用 GAP+MLP 得到一个标量权重，做 softmax 归一化后对分支特征加权求和。
    - 再与投影特征 concat，1x1 fuse 回到原始通道 c，并残差到输入。
    形状约定：
      x:       [B, c,  H,  W]
      x0:      [B, c_red, H, W]
      feats:   M 条分支，每条 [B, c_red, H, W]
      alpha:   [B, M]
      z_sum:   [B, c_red, H, W]
      out:     [B, c, H, W]
    """
    def __init__(self, c: int, c_red: int | None = None, dilations=(1, 3, 5)):
        super().__init__()
        self.dils = tuple(dilations)
        c_red = c // 2 if c_red is None else c_red

        # 1x1 投影到压缩通道
        self.proj = nn.Sequential(
            nn.Conv2d(c, c_red, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_red),
            nn.SiLU(inplace=True),
        )

        # 共享 3x3 核（不带偏置），不同分支只改变 dilation/padding
        self.shared = nn.Conv2d(c_red, c_red, kernel_size=3, stride=1, padding=1, bias=False)

        # 每个分支一个 MLP（GAP -> MLP -> 标量），用于 softmax 选通
        hid = max(c_red // 4, 16)
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c_red, hid, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hid, 1, kernel_size=1, bias=True)
            )
            for _ in self.dils
        ])

        # 1x1 融合回原通道，再残差
        self.out = nn.Sequential(
            nn.Conv2d(c_red * 2, c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def _same_padding_for_k3(d: int) -> int:
        # 对 3x3 卷积，“same” 对齐的 padding = dilation
        return d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 投影
        x0 = self.proj(x)  # [B, c_red, H, W]

        # 共享核做不同 dilation 的分支
        feats = []
        alphas = []
        weight = self.shared.weight
        for d, mlp in zip(self.dils, self.mlps):
            z = F.conv2d(x0, weight=weight, bias=None,
                         stride=1, padding=self._same_padding_for_k3(d), dilation=d)
            feats.append(z)  # [B, c_red, H, W]
            alphas.append(mlp(F.adaptive_avg_pool2d(z, 1)))  # [B, 1, 1, 1]

        # 分支权重 softmax
        # alphas_cat: [B, M, 1, 1] -> squeeze 到 [B, M]
        alphas_cat = torch.cat(alphas, dim=1)                     # [B, M, 1, 1]
        alpha = torch.softmax(alphas_cat.squeeze(-1).squeeze(-1), dim=1)  # [B, M]

        # 对 M 条分支按 alpha 加权求和
        feats_stacked = torch.stack(feats, dim=1)                 # [B, M, c_red, H, W]
        alpha_ = alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, M, 1, 1, 1]
        z_sum = (alpha_ * feats_stacked).sum(dim=1)               # [B, c_red, H, W]

        # 融合 + 残差
        y = self.out(torch.cat([x0, z_sum], dim=1))               # [B, c, H, W]
        return x + y
# ---------- Detector ----------
@register()
class DEIM(nn.Module):
    __inject__ = ["backbone", "encoder", "decoder"]

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module,
                 c4_in=512, c5_in=1024, c_mid=256,
                 use_ilda: bool = True, use_abf: bool = True, use_wsdc: bool = True):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

        self.use_ilda = use_ilda
        self.use_abf  = use_abf
        self.use_wsdc = use_wsdc

        # --- ILDA 或 P4 对齐兜底（二选一创建，保证 Params 按开关变动） ---
        if self.use_ilda:
            self.ilda = ILDA(c4_in=c4_in, c5_in=c5_in, c_mid=c_mid)
            self.align4 = None
        else:
            self.ilda = None
            self.align4 = Conv(c4_in, c_mid, k=1, act=True)

        # --- ABF 或 P5 对齐兜底（二选一创建） ---
        if self.use_abf:
            # ABF2 里会把 P5 对齐成 c_mid，要求输入 P4 已是 c_mid
            self.abf = ABF2(c_in4=c_mid, c_in5=c5_in, c_mid=c_mid)
            self.align5 = None
        else:
            self.abf = None
            self.align5 = Conv(c5_in, c_mid, k=1, act=True)

        # --- WSDC 可选创建（关就不占参数） ---
        if self.use_wsdc:
            self.wsdc4 = WSDC(c_mid)
            self.wsdc5 = WSDC(c_mid)
        else:
            self.wsdc4 = nn.Identity()
            self.wsdc5 = nn.Identity()

    def forward(self, x, targets=None):
        feats = self.backbone(x)
        assert len(feats) >= 2, "Backbone must return at least two levels (P4,P5)"
        p4, p5 = feats[-2], feats[-1]

        # ILDA or P4 1x1 align
        if self.ilda is not None:
            p4_cmid = self.ilda(p4, p5)
        else:
            p4_cmid = self.align4(p4)

        # ABF or P5 1x1 align
        if self.abf is not None:
            p4_fused, p5_fused = self.abf(p4_cmid, p5)
        else:
            p4_fused, p5_fused = p4_cmid, self.align5(p5)

        # 同层上下文（可能是 Identity）
        p4_ctx = self.wsdc4(p4_fused)
        p5_ctx = self.wsdc5(p5_fused)

        x = self.encoder([p4_ctx, p5_ctx])
        x = self.decoder(x, targets)
        return x
    def deploy(self):
        # 如果没有可重参数化模块，这里就是 no-op；但保留以兼容统计/导出流程
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self

