"""Generate the 200-epoch matched arms for the footprint training stage into configs/train-div2k/footprint/."""
import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'configs' / 'train-div2k' / 'footprint'
base = yaml.safe_load((ROOT / 'configs/train-div2k/train-liif-edsr-200ep.yaml').read_text())
lte_model = yaml.safe_load((ROOT / 'configs/train-div2k/train-lte.yaml').read_text())['model']
sc_model = yaml.safe_load((ROOT / 'configs/train-div2k/train-sc-inr.yaml').read_text())['model']
nosinc_model = yaml.safe_load((ROOT / 'configs/train-div2k/train-sc-inr-nosinc.yaml').read_text())['model']
enc = {'name': 'edsr-baseline', 'args': {'no_upsampling': True}}


def arm(model, asym=False):
    c = copy.deepcopy(base)
    c['model'] = copy.deepcopy(model)
    if asym:
        c['train_dataset']['wrapper']['name'] = 'sr-implicit-downsampled-asym'
    return c


arms = {
    'lte-200ep': arm(lte_model),
    'sc-inr-200ep': arm(sc_model),
    'sc-inr-nosinc-200ep': arm(nosinc_model),
    'liif-asym-200ep': arm(base['model'], asym=True),
    'lte-asym-200ep': arm(lte_model, asym=True),
    'sc-inr-exact-200ep': arm({'name': 'sc_inr_exact', 'args': {'encoder_spec': enc, 'num_freqs': 128, 'num_angles': 8,
                                                                 'freq_min': 0.1, 'freq_max': 2.0, 'omega_bound': 2.1,
                                                                 'coef_hidden': [128], 'learn_phase': True, 'base': 'bilinear'}}),
    'lte-linear-200ep': arm({'name': 'lte_linear', 'args': {'encoder_spec': enc, 'hidden_dim': 256}}),
}
OUT.mkdir(parents=True, exist_ok=True)
for name, cfg in arms.items():
    (OUT / f'train-{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
print(sorted(arms))
