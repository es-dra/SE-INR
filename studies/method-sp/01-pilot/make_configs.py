import yaml, copy
c = yaml.safe_load(open('configs/train-div2k/train-liif-edsr-200ep.yaml'))
for d in ('train_dataset', 'val_dataset'):
    c[d]['wrapper']['name'] = 'sr-implicit-downsampled-window'
    c[d]['wrapper']['args'].pop('sample_q', None)
    c[d]['wrapper']['args']['window'] = 48
arms = {}
arms['base'] = copy.deepcopy(c)
b = copy.deepcopy(c); b['model']['name'] = 'liif_sp'; arms['film'] = b
u = copy.deepcopy(c); u['freq_loss'] = {'weight': 0.5, 'u_min': 0.75}; arms['uloss'] = u
f = copy.deepcopy(b); f['freq_loss'] = {'weight': 0.5, 'u_min': 0.75}; arms['full'] = f
w = copy.deepcopy(c); w['model']['args']['imnet_spec']['args']['hidden_list'] = [316] * 4; arms['wide'] = w
for k, v in arms.items(): yaml.safe_dump(v, open(f'../train-{k}.yaml', 'w'), sort_keys=False)
print(list(arms))
