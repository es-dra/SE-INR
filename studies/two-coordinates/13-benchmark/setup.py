import yaml, os
src = open('/workspace/SE-INR/SE-INR-current/entrypoints/test.py').read()
old = "        model.gen_feat(inp)"
assert src.count(old) == 1
src = src.replace(old, "        model.gen_feat(inp)\n        model.feat = model.feat[:, :, :inp.shape[-2], :inp.shape[-1]]  # crop encoders that pad (SwinIR)")
open('test_crop.py', 'w').write(src)
base = yaml.safe_load(open('/workspace/SE-INR/SE-INR-current/configs/test/test-set14-4.yaml'))
os.makedirs('cfg', exist_ok=True)
for ds, root in [('set5', 'Set5'), ('set14', 'Set14'), ('bsd100', 'BSD100'), ('urban100', 'Urban100')]:
    for s in (2, 3, 4, 6):
        c = yaml.safe_load(yaml.safe_dump(base))
        c['test_dataset']['dataset']['args']['root_path'] = f'/workspace/SE-INR/Data/{root}/HR'
        c['test_dataset']['wrapper']['args'].update(scale_min=s, scale_max=s)
        c['eval_type'] = f'benchmark-{s}'
        yaml.safe_dump(c, open(f'cfg/{ds}-{s}.yaml', 'w'))
print('ok')
