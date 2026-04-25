""" Train for generating LIIF, from image to implicit representation.

    Config:
        train_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        val_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        (data_norm):
            inp: {sub: []; div: []}
            gt: {sub: []; div: []}
        (eval_type):
        (eval_bsize):

        model: $spec
        optimizer: $spec
        epoch_max:
        (multi_step_lr):
            milestones: []; gamma: 0.5
        (cosine_lr):
            T_max: ; eta_min:
        (warmup_cosine_lr):
            warmup_steps: ; eta_min:
        (lambda_cons): float (default 0.0)
        (grad_clip): float (default None)
        (resume): *.pth

        (epoch_val): ; (epoch_save):
"""


import warnings
warnings.filterwarnings("ignore")


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=2, persistent_workers=True, pin_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    Im_loader = make_data_loader(config.get('Im_dataset'), tag='Imaging')
    return train_loader, val_loader, Im_loader


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, eta_min=0):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(eta_min / optimizer.defaults['lr'],
                   0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def prepare_training(train_loader=None):
    epoch_max = config['epoch_max']
    step_scheduler = False

    if config.get('resume') is not None:
        resume_cfg = config['resume']
        if isinstance(resume_cfg, dict):
            sv_file = torch.load(resume_cfg['path'],
                                 map_location=torch.device(resume_cfg.get('map_location', 'cuda')))
        else:
            sv_file = torch.load(resume_cfg)
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('warmup_cosine_lr') is not None and train_loader is not None:
            warmup_cfg = config['warmup_cosine_lr']
            warmup_steps = warmup_cfg['warmup_steps']
            eta_min = warmup_cfg.get('eta_min', 1e-7)
            total_steps = epoch_max * len(train_loader)
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer, warmup_steps, total_steps, eta_min)
            step_scheduler = True
        elif config.get('cosine_lr') is not None:
            lr_scheduler = CosineAnnealingLR(optimizer, **config['cosine_lr'])
        elif config.get('multi_step_lr') is not None:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        else:
            lr_scheduler = None
        if not step_scheduler:
            for _ in range(epoch_start - 1):
                lr_scheduler.step() if lr_scheduler else None
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('warmup_cosine_lr') is not None and train_loader is not None:
            warmup_cfg = config['warmup_cosine_lr']
            warmup_steps = warmup_cfg['warmup_steps']
            eta_min = warmup_cfg.get('eta_min', 1e-7)
            total_steps = epoch_max * len(train_loader)
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer, warmup_steps, total_steps, eta_min)
            step_scheduler = True
        elif config.get('cosine_lr') is not None:
            lr_scheduler = CosineAnnealingLR(optimizer, **config['cosine_lr'])
        elif config.get('multi_step_lr') is not None:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        else:
            lr_scheduler = None

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler, step_scheduler


def train(train_loader, model, optimizer, lr_scheduler=None, step_scheduler=False,
          lambda_cons=0.0, grad_clip=None, use_amp=False):
    model.train()
    loss_fn = nn.L1Loss()
    train_loss = utils.Averager()

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        gt = (batch['gt'] - gt_sub) / gt_div

        with torch.amp.autocast('cuda', enabled=use_amp):
            if lambda_cons > 0:
                pred, y_eq1 = model(inp, batch['coord'], batch['cell'], return_eq=True)
                loss_main = loss_fn(pred, gt)

                B = batch['cell'].shape[0]
                cons_ratio = torch.rand(B, 1, 1, device=batch['cell'].device) * 1.5 + 0.5

                with torch.no_grad():
                    _, y_eq2 = model(inp, batch['coord'], batch['cell'],
                                     return_eq=True, cons_ratio=cons_ratio)

                loss_cons = F.mse_loss(y_eq1, y_eq2.detach())
                loss = loss_main + lambda_cons * loss_cons
            else:
                pred = model(inp, batch['coord'], batch['cell'])
                loss = loss_fn(pred, gt)

        train_loss.add(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if step_scheduler and lr_scheduler is not None:
            lr_scheduler.step()

        pred = None; loss = None

    return train_loss.item()


def main(config_, save_path, args=None):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    global config, log
    config = config_
    log = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader, Im_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler, step_scheduler = prepare_training(train_loader)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    lambda_cons = config.get('lambda_cons', 0.0)
    grad_clip = config.get('grad_clip', None)

    timer = utils.Timer()
    currEpoch = 0
    if args.show_tempImage:
        args.save_path = save_path
        if not os.path.exists(save_path+'/temp_Images'):
            os.mkdir(save_path+'/temp_Images')
            print('mkdir', save_path+'/temp_Images')

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        if args.show_tempImage:
            eval_psnr(Im_loader, model,
                data_norm=config['data_norm'],
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'),
                window_size = config.get('window_size'),
                args = args, ImNumber = epoch-1)

        if epoch == 1:
            if n_gpus > 1:
                model_ = model.module
            else:
                model_ = model
            model_spec = config['model']
            model_spec['sd'] = model_.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()
            sv_file = {
                'model': model_spec,
                'optimizer': optimizer_spec,
                'epoch': epoch
            }

            model.train()
            torch.save(sv_file, os.path.join(save_path, 'epoch-ini.pth'))


        train_loss = train(train_loader, model, optimizer,
                           lr_scheduler=lr_scheduler, step_scheduler=step_scheduler,
                           lambda_cons=lambda_cons, grad_clip=grad_clip,
                           use_amp=config.get('use_amp', False))
        if not step_scheduler and lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        model.train()
        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
            val_res = eval_psnr(val_loader, model_,
                data_norm=config['data_norm'],
                eval_type=None,
                eval_bsize=config.get('eval_bsize'))

            log_info.append('val: psnr={:.4f}'.format(val_res))
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))


if __name__ == '__main__':

    import argparse
    import os
    import math

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default = 'configs/train-div2k/train_edsr-baseline-liif.yaml' )
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default='1')
    parser.add_argument('--device', default='0')
    parser.add_argument('--show_tempImage', action='store_true')
    parser.add_argument('--saveFolder', default='./save')

    args = parser.parse_args()

    import sys
    if '--device' in sys.argv:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    import yaml
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, LambdaLR

    import datasets
    import models
    import utils
    from test import eval_psnr


    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None and args.tag != '':
        save_name += '_' + args.tag
    save_path = os.path.join(args.saveFolder, save_name)

    main(config, save_path, args)
