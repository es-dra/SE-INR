"""
SE-INR: Stage 0 - Log-Frequency Positional Encoding

仅替换 LIIF 的标准 Fourier PE 为 Log-Freq PE，MLP decoder 完全不变。
目标：验证 Log-Freq PE 本身不损害性能。

核心设计（按 Doc1_SE_INR_Architecture_Plan）：
- K=24 频率通道，对数等距分布 ω_k = ω_min * exp(k * Δω)
- ω_min=1.0, ω_max=64.0
- 对 2D 坐标 δ = (δ_x, δ_y) 编码为 [K×4]：每频率 4 分量 [sin(ω_k·δ_x), cos(ω_k·δ_x), sin(ω_k·δ_y), cos(ω_k·δ_y)]
- MLP 输入改为 [feat, log_freq_pe(rel_coord), cell]，MLP 结构不变
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord


class LogFreqPE(nn.Module):
    """
    对数频率位置编码器。

    频率集合：ω_k = ω_min * exp(k * Δω), k = 0, 1, ..., K-1
    其中 Δω = log(ω_max / ω_min) / (K - 1)

    对 2D 坐标 δ = (δ_x, δ_y) 编码为：
        [sin(ω_0·δ_x), cos(ω_0·δ_x), sin(ω_0·δ_y), cos(ω_0·δ_y),
         sin(ω_1·δ_x), cos(ω_1·δ_x), sin(ω_1·δ_y), cos(ω_1·δ_y),
         ...,
         sin(ω_{K-1}·δ_x), cos(ω_{K-1}·δ_x), sin(ω_{K-1}·δ_y), cos(ω_{K-1}·δ_y)]

    输出形状：[B, Q, K, 4] → reshape 为 [B, Q, K*4]
    """

    def __init__(self, K=24, omega_min=1.0, omega_max=64.0):
        super().__init__()
        self.K = K
        self.omega_min = omega_min
        self.omega_max = omega_max

        # 预计算对数等距频率（无可学习参数）
        delta_omega = (math.log(omega_max) - math.log(omega_min)) / (K - 1)
        self.register_buffer(
            'frequencies',
            torch.tensor([omega_min * math.exp(k * delta_omega) for k in range(K)])
        )

    def forward(self, rel_coord):
        """
        rel_coord: [B, Q, 2] 相对坐标，归一化到 [-1, 1]
        返回:     [B, Q, K*4] 对数频率 PE
        """
        B, Q = rel_coord.shape[:2]
        freq = self.frequencies.view(1, 1, self.K)           # [1, 1, K]

        delta_x = rel_coord[:, :, 0:1]                        # [B, Q, 1]
        delta_y = rel_coord[:, :, 1:2]                        # [B, Q, 1]

        # [B, Q, K, 1] * [1, 1, K] = [B, Q, K]
        wx = (delta_x * freq).unsqueeze(-1)                   # [B, Q, K, 1]
        wy = (delta_y * freq).unsqueeze(-1)                   # [B, Q, K, 1]

        pe = torch.cat([
            torch.sin(wx), torch.cos(wx),
            torch.sin(wy), torch.cos(wy)
        ], dim=-1)                                             # [B, Q, K, 4]

        return pe.view(B, Q, self.K * 4)                       # [B, Q, K*4]


import math


@register('se-inr')
class SEINR(nn.Module):
    """
    Stage 0 实现：Log-Freq PE + 标准 LIIF MLP

    仅替换 rel_coord 的编码方式（2D → K*4D Log-Freq PE），
    MLP decoder 结构与 LIIF 完全一致。
    """

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True,
                 K=24, omega_min=1.0, omega_max=64.0):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)

        # Log-Freq PE 模块（Stage 0 的核心改动）
        self.log_freq_pe = LogFreqPE(K=K, omega_min=omega_min, omega_max=omega_max)
        self.K = K
        pe_out_dim = K * 4  # 96 for K=24

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 9
            imnet_in_dim += pe_out_dim   # 用 Log-Freq PE 替换标准 2D 坐标
            if self.cell_decode:
                imnet_in_dim += 2
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).to(feat.device) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                # === 核心改动：用 Log-Freq PE 替换标准 2D 坐标 ===
                pe = self.log_freq_pe(rel_coord)              # [B, Q, K*4]

                inp = torch.cat([q_feat, pe], dim=-1)         # [B, Q, feat+pe_dim]

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
