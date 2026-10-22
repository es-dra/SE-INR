"""Intervene on LIIF cell only; preserve architecture and training data RNG."""
import json, os
from pathlib import Path
import torch
from models import register
from models.liif import LIIF

# w_hr = round(48*s), s ~ Uniform[1,4]. Endpoint bins have half mass.
CONSTANT_CELL = sum((0.5 if k in (48, 192) else 1.0) * 96.0 / k for k in range(48, 193)) / 144.0


def derangement(n, generator):
    if n < 2:
        raise ValueError('Shuffle arm requires batch size >= 2; singleton is not shuffled.')
    while True:
        p = torch.randperm(n, generator=generator)
        if bool((p != torch.arange(n)).all()):
            return p


@register('liif-cell-source')
class CellSourceLIIF(LIIF):
    def __init__(self, cell_mode, cell_seed, **kwargs):
        super().__init__(**kwargs)
        if cell_mode not in ('constant', 'shuffle') or not self.cell_decode:
            raise ValueError('Expected constant/shuffle with two cell dimensions retained')
        self.cell_mode = cell_mode
        self.cell_seed = int(cell_seed)
        self.cell_generator = torch.Generator(device='cpu').manual_seed(self.cell_seed)
        self.eval_cell_mode = 'native'
        self.eval_cell_value = CONSTANT_CELL
        self.audit_batches = 0
        self.audit_diff_sum = 0.0
        self.audit_native_sum = self.audit_assigned_sum = 0.0
        self.audit_native_sq = self.audit_assigned_sq = self.audit_product_sum = 0.0
        self.audit_n = 0

    def fixed_cell(self, cell, value):
        c = torch.empty_like(cell)
        c[..., 0] = value / self.feat.shape[-2]
        c[..., 1] = value / self.feat.shape[-1]
        return c

    def train(self, mode=True):
        result = super().train(mode)
        if not mode and self.audit_n and os.environ.get('CELL_SOURCE_AUDIT_PATH'):
            n = self.audit_n
            mx, my = self.audit_native_sum/n, self.audit_assigned_sum/n
            vx = self.audit_native_sq/n-mx*mx
            vy = self.audit_assigned_sq/n-my*my
            corr = (self.audit_product_sum/n-mx*my)/(max(vx*vy, 0)**.5) if vx*vy > 1e-20 else None
            report = dict(mode=self.cell_mode, batches=self.audit_batches, samples=n,
                          fixed_index_count=0 if self.cell_mode=='shuffle' else None,
                          same_cell_value_fraction=1-self.audit_diff_sum/n,
                          native_mean=mx, assigned_mean=my, native_variance=vx,
                          assigned_variance=vy, correlation=corr,
                          constant_expected=CONSTANT_CELL,
                          exact_marginal_sum_error=self.audit_native_sum-self.audit_assigned_sum,
                          exact_marginal_square_error=self.audit_native_sq-self.audit_assigned_sq)
            Path(os.environ['CELL_SOURCE_AUDIT_PATH']).write_text(json.dumps(report, indent=2)+'\n')
        return result

    def query_rgb(self, coord, cell=None):
        native = cell
        if self.cell_mode == 'constant':
            cell = self.fixed_cell(cell, CONSTANT_CELL)
        elif self.training:
            perm = derangement(cell.shape[0], self.cell_generator).to(cell.device)
            assert bool((perm != torch.arange(cell.shape[0], device=cell.device)).all())
            cell = cell.index_select(0, perm)
        elif self.eval_cell_mode == 'mean':
            cell = self.fixed_cell(cell, CONSTANT_CELL)
        elif self.eval_cell_mode == 'marginal':
            cell = self.fixed_cell(cell, self.eval_cell_value)
        elif self.eval_cell_mode != 'native':
            raise ValueError(self.eval_cell_mode)
        if self.training and self.audit_batches < 1000:
            x = native[:, 0, 0].detach().double() * self.feat.shape[-2]
            y = cell[:, 0, 0].detach().double() * self.feat.shape[-2]
            self.audit_diff_sum += float((x != y).sum().item())
            self.audit_native_sum += float(x.sum().item())
            self.audit_assigned_sum += float(y.sum().item())
            self.audit_native_sq += float(x.square().sum().item())
            self.audit_assigned_sq += float(y.square().sum().item())
            self.audit_product_sum += float((x*y).sum().item())
            self.audit_n += x.numel()
            self.audit_batches += 1
        return super().query_rgb(coord, cell)
