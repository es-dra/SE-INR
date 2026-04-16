import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SE_Linear_Input(nn.Module):
    def __init__(self, inFeat_dim, outNum, scales=None,
                 coord_scale=1.0, bias=True, iniScale=1.0):
        super().__init__()
        if scales is None:
            scales = [1.0, 1.414, 2.0, 2.828, 4.0]
        self.scales = torch.tensor(scales, dtype=torch.float32)
        self.num_scales = len(scales)
        self.inFeat_dim = inFeat_dim
        self.outNum = outNum
        self.coord_scale = coord_scale

        self.weight = nn.Parameter(
            torch.Tensor(outNum, inFeat_dim + 2, self.num_scales),
            requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(outNum, self.num_scales))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        BQ = input.shape[0]
        feat = input[:, :-2].reshape(BQ, self.inFeat_dim, self.num_scales)
        coord = input[:, -2:] * self.coord_scale

        scales_dev = self.scales.to(input.device)

        scaled_coords = []
        for k in range(self.num_scales):
            delta_k = coord / scales_dev[k]
            scaled_coords.append(delta_k)
        stacked_coords = torch.stack(scaled_coords, dim=2)

        paired = torch.cat([feat, stacked_coords], dim=1)

        out_list = []
        for k in range(self.num_scales):
            out_k = F.linear(paired[:, :, k], self.weight[:, :, k])
            out_list.append(out_k)
        out = torch.stack(out_list, dim=2)

        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)

        return out


class SE_Linear_Inter(nn.Module):
    def __init__(self, inNum, outNum, num_scales=5,
                 kernel_size=3, bias=True, iniScale=1.0):
        super().__init__()
        self.num_scales = num_scales
        self.inNum = inNum
        self.outNum = outNum
        padding = kernel_size // 2

        self.conv = nn.Conv1d(
            inNum, outNum, kernel_size,
            padding=padding, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
        if self.conv.bias is not None:
            fan_in = self.conv.in_channels
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.conv.bias, -bound, bound)

    def forward(self, input):
        BQ, C, K = input.shape
        x = self.conv(input)
        return x


class SE_Linear_Output_Uniform(nn.Module):
    def __init__(self, inNum, outNum, num_scales=5,
                 bias=True, iniScale=1.0):
        super().__init__()
        self.num_scales = num_scales
        self.inNum = inNum
        self.outNum = outNum

        self.weight = nn.Parameter(torch.Tensor(outNum, inNum), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(outNum))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        BQ, C, K = input.shape
        pooled = input.mean(dim=2)
        out = F.linear(pooled, self.weight, self.bias)
        return out


class SE_Linear_Output_Weighted(nn.Module):
    def __init__(self, inNum, outNum, num_scales=5,
                 scales=None, sigma=0.5, bias=True, iniScale=1.0):
        super().__init__()
        self.num_scales = num_scales
        self.inNum = inNum
        self.outNum = outNum
        self.sigma = sigma

        if scales is None:
            scales = [1.0, 1.414, 2.0, 2.828, 4.0]
        self.register_buffer('log_scales',
                             torch.log(torch.tensor(scales, dtype=torch.float32)))

        self.weight = nn.Parameter(torch.Tensor(outNum, inNum), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(outNum))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, target_scale_log=None):
        BQ, C, K = input.shape

        if target_scale_log is not None:
            ts = target_scale_log.view(-1, 1)
            dist_sq = (self.log_scales.view(1, -1) - ts) ** 2
            alpha = F.softmax(-dist_sq / (2 * self.sigma ** 2), dim=1)
            alpha = alpha.unsqueeze(1)
            weighted = (input * alpha).sum(dim=2)
        else:
            weighted = input.mean(dim=2)

        out = F.linear(weighted, self.weight, self.bias)
        return out


class SE_Linear_Output_Nearest(nn.Module):
    def __init__(self, inNum, outNum, num_scales=5,
                 scales=None, bias=True, iniScale=1.0):
        super().__init__()
        self.num_scales = num_scales
        self.inNum = inNum
        self.outNum = outNum

        if scales is None:
            scales = [1.0, 1.414, 2.0, 2.828, 4.0]
        self.register_buffer('scales',
                             torch.tensor(scales, dtype=torch.float32))

        self.weight = nn.Parameter(torch.Tensor(outNum, inNum), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(outNum))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, target_scale_val=None):
        BQ, C, K = input.shape

        if target_scale_val is not None:
            dist = torch.abs(self.scales.view(1, -1) - target_scale_val.view(-1, 1))
            idx = dist.argmin(dim=1)
            selected = input[torch.arange(BQ, device=input.device), :, idx]
        else:
            K_mid = K // 2
            selected = input[:, :, K_mid]

        out = F.linear(selected, self.weight, self.bias)
        return out


class SE_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list, num_scales=5,
                 scales=None, output_mode='uniform',
                 inter_kernel_size=3, coord_scale=1.0,
                 use_cell_decode=True, flat_mode=False,
                 num_se_layers=2, **kwargs):
        """
        num_se_layers: 保留几层 SE_Linear_Inter 再池化 (默认2)
        """
        super().__init__()
        self.num_scales = num_scales
        self.output_mode = output_mode
        self.use_cell_decode = use_cell_decode
        self.flat_mode = flat_mode
        self.num_se_layers = num_se_layers

        if scales is None:
            scales = _default_scale_set(num_scales)
        self.scales = scales

        if flat_mode:
            layers = []
            prev_dim = in_dim
            for hdim in hidden_list:
                layers.append(nn.Linear(prev_dim, hdim))
                layers.append(nn.ReLU(inplace=True))
                prev_dim = hdim
            self.mlp_body = nn.Sequential(*layers)
            self.output_layer = nn.Linear(prev_dim, out_dim)
            self.cell_mlp = None
        else:
            cell_dim = 2 * num_scales if use_cell_decode else 0
            feat_total = in_dim - 2 - cell_dim
            feats_per_scale = feat_total // num_scales
            actual_in = feats_per_scale

            # === 完整 SE_MLP: 保留所有 SE 层 ===
            se_layers = []
            prev_dim = actual_in

            # Input layer (equvariant)
            se_layers.append(SE_Linear_Input(prev_dim, hidden_list[0],
                                              scales=scales,
                                              coord_scale=coord_scale))
            prev_dim = hidden_list[0]

            # SE_Linear_Inter layers (保留 num_se_layers 层)
            for i in range(num_se_layers):
                se_layers.append(SE_Linear_Inter(prev_dim, hidden_list[0],
                                                  num_scales,
                                                  kernel_size=inter_kernel_size))
                se_layers.append(nn.ReLU(inplace=True))

            self.se_layers = nn.ModuleList(se_layers)

            # === 中间层处理后池化 ===
            self.scale_pool = lambda x: x.mean(dim=2, keepdim=True)

            # === 池化后使用标准 MLP ===
            mlp_layers = []
            for hdim in hidden_list[1:]:
                mlp_layers.append(nn.Linear(prev_dim, hdim))
                mlp_layers.append(nn.ReLU(inplace=True))
                prev_dim = hdim
            self.mlp_body = nn.Sequential(*mlp_layers)
            self.output_layer = nn.Linear(prev_dim, out_dim)

            # Cell MLP
            if use_cell_decode:
                self.cell_mlp = nn.Sequential(
                    nn.Linear(2 * num_scales, num_scales),
                    nn.ReLU(inplace=True),
                    nn.Linear(num_scales, prev_dim))

    def forward(self, input, target_scale_log=None, target_scale_val=None):
        if self.flat_mode:
            x = input
            x = self.mlp_body(x)
            return self.output_layer(x)

        BQ = input.shape[0]
        feat_coord_dim = self.se_layers[0].inFeat_dim * self.num_scales + 2
        cell_dim = 2 * self.num_scales if self.use_cell_decode else 0

        x_feat_coord = input[:, :feat_coord_dim]
        x = x_feat_coord

        # SE layers (equvariant input)
        for layer in self.se_layers:
            x = layer(x)

        # === Pool after SE layers ===
        x = self.scale_pool(x)

        # Cell decode after pool
        if self.use_cell_decode and self.cell_mlp is not None:
            x_cell = input[:, feat_coord_dim:]
            cell_feat = self.cell_mlp(x_cell)
            cell_feat = cell_feat.reshape(BQ, -1, 1)
            x = x + cell_feat

        # MLP body after pool
        x = x.squeeze(2)  # [B, hidden]
        x = self.mlp_body(x)
        out = self.output_layer(x)

        return out


def _default_scale_set(num_scales=5):
    if num_scales == 5:
        return [1.0, 1.414, 2.0, 2.828, 4.0]
    elif num_scales == 4:
        return [round(2 ** (i / 3), 4) for i in range(4)]
    elif num_scales == 3:
        return [1.0, 2.0, 4.0]
    elif num_scales == 7:
        base = 2 ** (1 / 6)
        return [round(base ** i, 4) for i in range(7)]
    else:
        base = 4 ** (1 / (num_scales - 1))
        return [round(base ** i, 4) for i in range(num_scales)]
