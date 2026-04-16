"""
SE-INR Stage 1: Log-Freq PE + Additive Injection + 1D Conv Decoder + Scale Readout

简化架构（2026-04-12）:
- Log-Freq PE: 对数等距频率编码 (同 Stage 0)
- Additive Injection: 加法注入替代 FiLM（轻量化）
- SE-Decoder: 沿频率轴 1D 卷积 (保持等变性)
- Scale Readout: 基于目标尺度插值读出

参照 Doc1_SE_INR_Architecture_Plan.md Section 3 + Section 8
简化方案参照 Doc1 Section 8.2
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord


class LogFreqPE(nn.Module):
    """
    对数频率位置编码器 (同 Stage 0)

    频率集合: ω_k = ω_min * exp(k * Δω), k = 0, 1, ..., K-1
    其中 Δω = log(ω_max / ω_min) / (K - 1)

    对 2D 坐标 δ = (δ_x, δ_y) 编码为 K×4 维:
    [sin(ω_0·δ_x), cos(ω_0·δ_x), sin(ω_0·δ_y), cos(ω_0·δ_y), ...]
    """
    def __init__(self, K=24, omega_min=1.0, omega_max=30.0):
        super().__init__()
        self.K = K
        self.omega_min = omega_min
        self.omega_max = omega_max

        delta_omega = (math.log(omega_max) - math.log(omega_min)) / (K - 1)
        self.register_buffer(
            'frequencies',
            torch.tensor([omega_min * math.exp(k * delta_omega) for k in range(K)])
        )
        self.delta_omega = delta_omega

    def forward(self, rel_coord):
        """
        rel_coord: [B, Q, 2] 相对坐标
        返回:       [B, Q, K, 4]
        """
        B, Q = rel_coord.shape[:2]
        freq = self.frequencies.view(1, 1, self.K)  # [1, 1, K]

        delta_x = rel_coord[:, :, 0:1]  # [B, Q, 1]
        delta_y = rel_coord[:, :, 1:2]  # [B, Q, 1]

        wx = (delta_x * freq).unsqueeze(-1)  # [B, Q, K, 1]
        wy = (delta_y * freq).unsqueeze(-1)  # [B, Q, K, 1]

        pe = torch.cat([
            torch.sin(wx), torch.cos(wx),
            torch.sin(wy), torch.cos(wy)
        ], dim=-1)  # [B, Q, K, 4]

        return pe


class AdditiveInjection(nn.Module):
    """
    加法注入层: 将 encoder 特征注入等变分支

    替代 FiLM 的轻量化方案:
    - FiLM: h = gamma * h_pe + beta (gamma/beta 频率常数)
    - 加法: h = pe_proj(pe) + z_proj(z)

    数学等价性:
    - FiLM: h_k = gamma * h_pe[k] + beta
    - 加法: h_k = W_pe * pe_k + W_z * z + b
    - 两者都是 线性(pe) + 线性(z) 的形式

    参照 Doc1 Section 8.2
    """
    def __init__(self, feat_dim=256, C_h=128):
        super().__init__()
        self.C_h = C_h
        # encoder 特征投影: feat_dim → C_h
        self.z_proj = nn.Linear(feat_dim, C_h)
        # PE 投影: 4 → C_h (无偏置，保持等变性)
        self.pe_proj = nn.Linear(4, C_h, bias=False)

    def forward(self, z, pe):
        """
        z:   [B*Q, feat_dim] encoder 特征
        pe:  [B, Q, K, 4]     Log-Freq PE
        返回: [B, Q, K, C_h]  融合后的特征
        """
        B, Q, K = pe.shape[:3]

        # PE 投影: [B, Q, K, 4] → [B*Q, K, C_h]
        pe_flat = pe.view(B * Q, K, 4)
        h_pe = self.pe_proj(pe_flat)  # [B*Q, K, C_h]

        # encoder 特征投影: [B*Q, feat_dim] → [B*Q, C_h]
        z_proj = self.z_proj(z)  # [B*Q, C_h]

        # 加法注入: z_proj 对所有频率位置相同 → 等变性不破坏
        h = h_pe + z_proj.unsqueeze(1)  # [B*Q, K, C_h]

        return h.view(B, Q, K, self.C_h)


class SEDecoderLayer(nn.Module):
    """
    SE-Decoder 单层: 沿频率轴的 1D 卷积 + 残差连接

    特性:
    - Conv1d 保持平移等变性
    - bias=False (定理4要求)
    - ReLU 逐元素激活 (不破坏等变性)
    - 残差连接 (恒等映射也是平移等变的)

    参照 Doc1 Section 3.6
    """
    def __init__(self, C_h, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            C_h, C_h, kernel_size,
            padding=padding, bias=False
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, h, res=None):
        """
        h:   [B, C_h, K]
        res: 可选的残差连接
        """
        out = self.act(self.conv(h))
        if res is not None:
            out = out + res
        return out


class SEDecoder(nn.Module):
    """
    SE-Decoder: 多层 1D 卷积沿频率轴

    3 层 Conv1d（简化方案，原为 4 层）:
    - K=24 的序列上，3 层 kernel=3 的感受野已覆盖 7 个频率位置

    参照 Doc1 Section 3.6 + Section 8.3
    """
    def __init__(self, C_h=128, num_layers=3, kernel_size=3):
        super().__init__()
        self.layers = nn.ModuleList([
            SEDecoderLayer(C_h, kernel_size)
            for _ in range(num_layers)
        ])

    def forward(self, h):
        """
        h: [B, C_h, K] 输入
        返回: [B, C_h, K] 输出
        """
        for layer in self.layers:
            h = layer(h) + h  # 残差连接
        return h


class ScaleReadoutIndex(nn.Module):
    """
    尺度读出层: 从频率轴插值读出 RGB (索引方式，向量化)

    输入:
    - h: [B, C_h, K] 等变层输出
    - cell: [B, 2] cell size (编码目标尺度), 同一 batch 内共享相同 scale
    - delta_omega: 对数频率间隔
    - tau_offset: cell 对数几何均值基线，将 cell 映射为尺度因子对数

    输出: [B, 3] RGB 预测

    tau 计算公式 (修正版，2026-04-16):
    - cell_geomean = sqrt(cell_x * cell_y) = 2 / crop_hr_size
    - crop_hr_size = inp_size * scale_factor
    - tau = (log(scale_factor) - log(scale_max)) / delta_omega
           = (-log(cell_geomean) - tau_offset) / delta_omega
    其中 tau_offset = -log(cell_geomean_min) = log(inp_size/2) = log(24) ≈ 3.178

    特性:
    - 不再使用 s.clamp() 截断，tau 自然落在 [0, K-1] 范围
    - omega_max=30 配合 delta_omega=log(30)/23 ≈ 0.148
    - 训练尺度 [1,4] 映射到 tau ∈ [0, 9.4]，OOD 尺度自然延伸到 [0, 23]

    参照 Doc1 Section 3.7 方案 A + 2026-04-16 修正
    """
    def __init__(self, C_h, K, delta_omega, tau_offset):
        super().__init__()
        self.C_h = C_h
        self.K = K
        self.delta_omega = delta_omega
        self.register_buffer('tau_offset', torch.tensor(tau_offset))
        self.to_rgb = nn.Linear(C_h, 3)

    def _compute_tau(self, cell):
        """计算 tau = (-log(cell_geomean) - tau_offset) / delta_omega"""
        cell_geomean = (cell[:, 0] * cell[:, 1]).sqrt().clamp(min=1e-10)
        log_cell_inv = -torch.log(cell_geomean)  # = log(scale_factor) + const
        tau = (log_cell_inv - self.tau_offset) / self.delta_omega
        return tau

    def forward(self, h, cell):
        """
        h:    [B, C_h, K]
        cell: [B, 2]
        返回: [B, 3]
        """
        B, C_h, K = h.shape

        tau = self._compute_tau(cell)
        k_low = tau.floor().long().clamp(0, K - 2)
        k_high = (k_low + 1).clamp(max=K - 1)
        alpha = (tau - k_low.float()).clamp(0.0, 1.0)

        h = h.permute(0, 2, 1)  # [B, K, C_h]
        batch_idx = torch.arange(B, device=h.device, dtype=torch.long)
        h_low = h[batch_idx, k_low]
        h_high = h[batch_idx, k_high]

        h_read = (1 - alpha.unsqueeze(-1)) * h_low + alpha.unsqueeze(-1) * h_high
        return self.to_rgb(h_read)

    def forward_vectorized(self, h, cell):
        """
        向量化版本: 处理 [B*Q, C_h, K] 输入 (所有 Q 点共享同一 scale)

        h:    [B*Q, C_h, K]
        cell: [B*Q, 2] 或 [B, 2] (broadcast)
        返回: [B*Q, 3]
        """
        BQ, C_h, K = h.shape

        tau = self._compute_tau(cell)
        k_low = tau.floor().long().clamp(0, K - 2)
        k_high = (k_low + 1).clamp(max=K - 1)
        alpha = (tau - k_low.float()).clamp(0.0, 1.0)

        h = h.permute(0, 2, 1)
        batch_idx = torch.arange(BQ, device=h.device, dtype=torch.long)
        h_low = h[batch_idx, k_low]
        h_high = h[batch_idx, k_high]

        h_read = (1 - alpha.unsqueeze(-1)) * h_low + alpha.unsqueeze(-1) * h_high
        return self.to_rgb(h_read)


@register('se-inr-s1')
class SEINRS1(nn.Module):
    """
    SE-INR Stage 1: 完整尺度等变 decoder

    架构（修正版，2026-04-16）:
    1. Log-Freq PE (无可学习参数，omega_max=30)
    2. Additive Injection (加法注入替代 FiLM)
    3. SE-Decoder (3层 Conv1d 沿频率轴)
    4. Scale Readout (尺度索引插值读出，-log(cell_geomean) 映射)

    关键修正 (2026-04-16):
    - omega_max: 64→30，使 delta_omega = log(30)/23 ≈ 0.148
    - tau_offset: log(24) ≈ 3.178，将 cell 正确映射到尺度因子对数
    - tau = (-log(cell_geomean) - tau_offset) / delta_omega
    - 训练尺度 [1,4] 映射到 tau ∈ [0, 9.4]，不再截断
    - C_h: 64→128，decoder 容量提升

    架构流程:
    1. Encoder (EDSR, 不修改)
    2. Feature Query (LIIF 方式, 不修改)
    3. Log-Freq PE (omega_max=30)
    4. Additive Injection
    5. SE-Decoder (3层 Conv1d)
    6. Scale Readout (新 tau 映射)

    参照 Doc1 Section 3.9 + Section 8.2 + Section 8.3
    """

    def __init__(self, encoder_spec,
                 K=24, omega_min=1.0, omega_max=30.0,
                 C_h=128, num_layers=3, kernel_size=3,
                 local_ensemble=True, feat_unfold=False,
                 inp_size=48):
        """
        参数:
        - K: 频率通道数 (默认 24)
        - omega_max: 对数频率上界 (默认 30.0，修正版)
        - C_h: 隐藏层维度 (默认 128)
        - num_layers: SE-Decoder 层数 (默认 3)
        - local_ensemble: 是否使用 local ensemble (默认 True, 与等变性正交)
        - feat_unfold: 是否使用 feature unfold (默认 False)
        - inp_size: 输入 patch 大小 (默认 48)，用于计算 tau_offset

        参数量估算 (C_h=128, num_layers=3):
        - z_proj (256→128): 33K
        - pe_proj (4→128): 512
        - 3层 Conv1d (128,128,3): 148K
        - to_rgb (128→3): 387
        - 合计: ~182K
        """
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.K = K
        self.C_h = C_h

        # Encoder (不修改)
        self.encoder = models.make(encoder_spec)

        # Log-Freq PE
        self.log_freq_pe = LogFreqPE(K=K, omega_min=omega_min, omega_max=omega_max)
        self.omega_min = omega_min
        delta_omega = (math.log(omega_max) - math.log(omega_min)) / (K - 1)
        self.delta_omega = delta_omega

        # tau_offset: log(inp_size/2)，将 cell 映射到尺度因子对数
        # omega_max=30, K=24 → delta_omega = log(30)/23 ≈ 0.148
        # tau = (-log(cell_geomean) - tau_offset) / delta_omega
        # tau_offset = log(inp_size/2) = log(24) ≈ 3.178
        tau_offset = math.log(inp_size / 2.0)

        # Additive Injection (替代 FiLM Fusion)
        feat_dim = self.encoder.out_dim
        if feat_unfold:
            feat_dim *= 9
        self.add_inj = AdditiveInjection(feat_dim, C_h)

        # SE-Decoder (3层 Conv1d)
        self.se_decoder = SEDecoder(C_h=C_h, num_layers=num_layers, kernel_size=kernel_size)

        # Scale Readout (传入 tau_offset)
        self.scale_readout = ScaleReadoutIndex(C_h, K, delta_omega, tau_offset)

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat

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
                    .permute(0, 2, 1)  # [B, Q, C]
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)  # [B, Q, 2]
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]  # [B, Q, 2]

                # === Stage 1 核心: SE-Decoder (简化版) ===
                # 1. Log-Freq PE
                pe = self.log_freq_pe(rel_coord)  # [B, Q, K, 4]

                # 2. Additive Injection (替代 FiLM)
                B, Q = coord.shape[:2]
                z = q_feat.reshape(B * Q, -1)  # [B*Q, C]
                h = self.add_inj(z, pe)         # [B, Q, K, C_h]

                # 3. SE-Decoder (1D conv along frequency)
                h = h.permute(0, 1, 3, 2)        # [B, Q, C_h, K]
                h = h.reshape(B * Q, self.C_h, self.K)  # [BQ, C_h, K]
                h = self.se_decoder(h)           # [BQ, C_h, K]

                # 4. Scale Readout (向量化)
                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]  # [B, Q, 2]
                cell_b = cell[:, 0, :]  # [B, 2], same scale for all Q in same batch
                cell_bq = cell_b.unsqueeze(1).expand(B, Q, 2).reshape(B * Q, 2)  # [BQ, 2]

                # Scale readout: h [BQ, C_h, K], cell [BQ, 2]
                rgb = self.scale_readout.forward_vectorized(h, cell_bq)  # [BQ, 3]
                rgb = rgb.view(B, Q, 3)  # [B, Q, 3]

                preds.append(rgb)

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
