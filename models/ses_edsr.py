import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from models import register
from models._sesn_imports import get_sesn_classes


def _default_scale_set(num_scales=5):
    if num_scales == 5:
        return [1.0, 1.414, 2.0, 2.828, 4.0]
    elif num_scales == 4:
        return [1.0, 2 ** (1 / 3), 2 ** (2 / 3), 2.0]
    elif num_scales == 7:
        base = 2 ** (1 / 6)
        [round(base ** i, 4) for i in range(7)]
    elif num_scales == 3:
        return [1.0, 2.0, 4.0]
    else:
        base = 4 ** (1 / (num_scales - 1))
        return [round(base ** i, 4) for i in range(num_scales)]


class SE_ResBlock(nn.Module):
    def __init__(self, n_feats_per_scale, num_scales, kernel_size,
                 effective_size=7, scales=None, bias=True,
                 act=nn.ReLU(True), res_scale=1, basis_type='A'):
        super().__init__()
        self.res_scale = res_scale
        sesn = get_sesn_classes()
        body = [
            sesn['SESConv_H_H'](n_feats_per_scale, n_feats_per_scale,
                                 scale_size=1, kernel_size=kernel_size,
                                 effective_size=effective_size,
                                 scales=scales, padding=kernel_size // 2,
                                 bias=bias, basis_type=basis_type),
            act,
            sesn['SESConv_H_H'](n_feats_per_scale, n_feats_per_scale,
                                 scale_size=1, kernel_size=kernel_size,
                                 effective_size=effective_size,
                                 scales=scales, padding=kernel_size // 2,
                                 bias=bias, basis_type=basis_type),
        ]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        return res + x


class StandardResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size=3, bias=True,
                 act=nn.ReLU(True), res_scale=1):
        super().__init__()
        self.res_scale = res_scale
        body = [
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2, bias=bias),
            act,
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2, bias=bias),
        ]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        return res + x


class SE_EDSR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = getattr(args, 'kernel_size', 3)
        act = nn.ReLU(True)
        res_scale = getattr(args, 'res_scale', 1)

        self.num_scales = getattr(args, 'num_scales', 5)
        self.scales = getattr(args, 'scales', None)
        if self.scales is None:
            self.scales = _default_scale_set(self.num_scales)

        self.num_se_layers = getattr(args, 'num_se_layers', n_resblocks)
        self.effective_size = getattr(args, 'effective_size', 7)
        self.basis_type = getattr(args, 'basis_type', 'A')

        K = self.num_scales
        feats_per_scale = n_feats // K
        assert feats_per_scale > 0, f'n_feats={n_feats} too small for K={K} scales'

        sesn = get_sesn_classes()

        in_dim = getattr(args, 'in_dim', 3)
        if getattr(args, 'cell_decode', False):
            in_dim = 3 * 3

        self.head_conv = sesn['SESConv_Z2_H'](
            in_dim, feats_per_scale, kernel_size,
            effective_size=self.effective_size,
            scales=self.scales, padding=kernel_size // 2,
            bias=True, basis_type=self.basis_type)

        se_blocks = []
        for _ in range(min(self.num_se_layers, n_resblocks)):
            se_blocks.append(
                SE_ResBlock(feats_per_scale, K, kernel_size,
                            effective_size=self.effective_size,
                            scales=self.scales, bias=True,
                            act=act, res_scale=res_scale,
                            basis_type=self.basis_type))

        std_blocks = []
        remaining = max(0, n_resblocks - self.num_se_layers)
        for _ in range(remaining):
            std_blocks.append(
                StandardResBlock(feats_per_scale * K, kernel_size,
                                  bias=True, act=act, res_scale=res_scale))

        self.se_body = nn.Sequential(*se_blocks) if se_blocks else nn.Identity()
        self.std_body = nn.Sequential(*std_blocks) if std_blocks else nn.Identity()
        self.has_std_body = len(std_blocks) > 0

        out_feats_per_scale = getattr(args, 'out_feats_per_scale', None) or feats_per_scale
        if self.has_std_body:
            self.tail_conv = nn.Conv2d(feats_per_scale * K, n_feats,
                                        kernel_size, padding=kernel_size // 2)
            # Projection layer to match head_res dimensions with tail_out
            self.head_proj = nn.Conv2d(feats_per_scale * K, n_feats, 1) if feats_per_scale * K != n_feats else nn.Identity()
            self._out_dim = n_feats
        else:
            self.tail_conv = sesn['SESConv_H_H'](
                feats_per_scale, out_feats_per_scale,
                scale_size=1, kernel_size=kernel_size,
                effective_size=self.effective_size,
                scales=self.scales, padding=kernel_size // 2,
                bias=True, basis_type=self.basis_type)
            self._out_dim = out_feats_per_scale * K

    def forward(self, x):
        B = x.shape[0]

        head_out = self.head_conv(x)

        body_out = self.se_body(head_out)

        if self.has_std_body:
            S = body_out.shape[2]
            Cps = body_out.shape[1]
            body_out_flat = body_out.permute(0, 2, 1, 3, 4).contiguous()
            body_out_flat = body_out_flat.view(B, -1, body_out.shape[3], body_out.shape[4])
            body_out = self.std_body(body_out_flat)
            tail_in = body_out
        else:
            tail_in = body_out

        tail_out = self.tail_conv(tail_in)

        if self.has_std_body:
            # Standard residual: head_out was collapsed to 4D
            head_res = head_out.permute(0, 2, 1, 3, 4).contiguous().view(
                B, -1, head_out.shape[3], head_out.shape[4])
            head_res = self.head_proj(head_res)
            res = tail_out + head_res
            # Convert 4D to 5D by adding scale dimension for SE-LIIF compatibility
            # [B, C, H, W] -> [B, C, 1, H, W]
            res = res.unsqueeze(2)
            return res
        else:
            # Scale-equivariant residual: project head_out channels to match tail_out
            # head_out: [B, feats_per_scale, K, H, W]
            # tail_out: [B, out_feats_per_scale, K, H, W] or [B, feats_per_scale, K, H, W]
            # Recalculate local vars for forward pass
            K = self.num_scales
            n_feats = self.args.n_feats
            feats_per_scale = n_feats // K
            out_feats_per_scale = getattr(self.args, 'out_feats_per_scale', None) or feats_per_scale
            if out_feats_per_scale == feats_per_scale:
                # Shapes match directly, no projection needed
                res = tail_out + head_out
            else:
                # Need to project: [B, feats_per_scale, K, H, W] -> [B, out_feats_per_scale, K, H, W]
                # Use 1x1 conv on flattened scale dim
                head_reshaped = head_out.permute(0, 2, 1, 3, 4).contiguous()  # [B, K, feats_per_scale, H, W]
                head_reshaped = head_reshaped.view(B * K, feats_per_scale, head_out.shape[3], head_out.shape[4])
                # Project each scale slice
                head_proj = nn.Conv2d(feats_per_scale, out_feats_per_scale, 1, bias=False)
                head_proj_out = head_proj(head_reshaped)  # [B*K, out_feats_per_scale, H, W]
                head_proj_out = head_proj_out.view(B, K, out_feats_per_scale, head_out.shape[3], head_out.shape[4])
                head_proj_out = head_proj_out.permute(0, 2, 1, 3, 4)  # [B, out_feats_per_scale, K, H, W]
                res = tail_out + head_proj_out
            return res

    @property
    def out_dim(self):
        return self._out_dim


@register('ses-edsr')
def make_ses_edsr(n_resblocks=16, n_feats=64, res_scale=1,
                  scale=2, no_upsampling=True, rgb_range=1,
                  kernel_size=3, cell_decode=False,
                  num_scales=5, scales=None, num_se_layers=None,
                  effective_size=7, basis_type='A',
                  out_feats_per_scale=None, in_dim=3):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale
    args.kernel_size = kernel_size
    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.rgb_range = rgb_range
    args.cell_decode = cell_decode
    args.num_scales = num_scales
    args.scales = scales
    args.num_se_layers = num_se_layers if num_se_layers is not None else n_resblocks
    args.effective_size = effective_size
    args.basis_type = basis_type
    args.out_feats_per_scale = out_feats_per_scale
    args.in_dim = in_dim
    model = SE_EDSR(args)
    model._out_dim = model.out_dim
    return model
