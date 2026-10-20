import yaml, re
for a, base in [('edsr', 'train-liif-edsr-200ep.yaml'), ('rdn', 'train-liif-rdn-200ep.yaml'), ('nocell', 'train-liif-edsr-nocell-200ep.yaml')]:
    c = yaml.safe_load(open(base))
    for d in ('train_dataset', 'val_dataset'): c[d]['wrapper']['name'] = 'sr-implicit-downsampled-gauss'
    yaml.safe_dump(c, open(f'train-gauss-{a}.yaml', 'w'), sort_keys=False)
src = open('run_v3.py').read()
old = "y = F.interpolate(torch.from_numpy(x.transpose(2, 0, 1))[None], size=(H//s, H//s), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)"
assert src.count(old) == 1
src = src.replace(old, "y = gauss_down(torch.from_numpy(x.transpose(2, 0, 1)), (H//s, H//s))[None]")
src = src.replace("sys.path.insert(0, A0.src); import models", "sys.path.insert(0, A0.src); import models; from datasets.gauss_wrapper import gauss_down")
open('run_v3_gauss.py', 'w').write(src)
open('src/datasets/__init__.py', 'a').write('\nfrom . import gauss_wrapper\n')
print('ok')
