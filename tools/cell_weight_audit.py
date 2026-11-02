#!/usr/bin/env python3
"""Audit the cell-reading weights of trained token-conditioned decoders (Proposition 1).

Under symmetric training (c_x = c_y) the gradient of the two cell columns is identical, so the
difference w_x - w_y keeps its initial value while the sum w_x + w_y is trained. For each checkpoint
this prints the RMS of both, their ratio, and the Kaiming-uniform initial RMS for that layer's fan-in,
so the trained difference can be compared with what initialisation alone predicts.

  python3 tools/cell_weight_audit.py /path/to/liif/epoch-last.pth /path/to/lte/epoch-last.pth ... [--json out.json]
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def kaiming_uniform_rms(fan_in):
    # torch.nn.Linear default: U(-b, b) with b = 1/sqrt(fan_in) (kaiming_uniform with a=sqrt(5)); RMS = b/sqrt(3)
    return 1 / math.sqrt(fan_in) / math.sqrt(3)


def audit(path):
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=True)
    name = ck['model']['name']
    sd = ck['model']['sd']
    rows = []
    if name in ('liif', 'liif_old'):
        W = sd['imnet.layers.0.weight']
        wx, wy = W[:, -2], W[:, -1]
        rows.append(('imnet.layers.0 (cell columns)', W.shape[1], wx, wy))
    elif name == 'lte':
        W = sd['phase.weight']
        rows.append(('phase (cell -> phase)', W.shape[1], W[:, 0], W[:, 1]))
    else:
        return {'path': str(path), 'registry_name': name, 'note': 'no linear cell input in this model'}
    out = {'path': str(path), 'registry_name': name, 'epoch': ck.get('epoch'), 'layers': []}
    for label, fan_in, wx, wy in rows:
        d, s = (wx - wy), (wx + wy)
        init = kaiming_uniform_rms(fan_in)
        out['layers'].append({'layer': label, 'fan_in': fan_in,
                              'rms_diff_w_x_minus_w_y': float(d.pow(2).mean().sqrt()),
                              'rms_sum_w_x_plus_w_y': float(s.pow(2).mean().sqrt()),
                              'rms_init_single_weight': init,
                              'rms_init_diff_expected': init * math.sqrt(2),
                              'rms_init_sum_expected': init * math.sqrt(2)})
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='+')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    res = [audit(p) for p in a.ckpts]
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + '\n')
