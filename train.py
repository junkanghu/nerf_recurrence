import os.path
import numpy as np
import imageio
import torch
from options.options import get_parser
from data import creat_dataset
from model import creat_nerf
from tqdm import tqdm, trange
from model.networks import *
import datetime
import sys


if __name__ == "__main__":
    begin = datetime.datetime.now()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')  # 所有定义的tensor全部送到cuda上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = get_parser()
    dataset = creat_dataset(opt)
    global_step, optimizer, train_kwargs, test_kwargs, result_dir = creat_nerf(opt)
    bds_dict = {'near': dataset['near'], 'far': dataset['far']}
    train_kwargs.update(bds_dict)
    test_kwargs.update(bds_dict)
    print('begin')
    print('View for train', dataset['i_train'], '\nView for validation', dataset['i_val'])
    # to tensor
    val_poses = torch.from_numpy(dataset['val_poses']).to(device)
    test_poses = torch.from_numpy(dataset['test_poses']).to(device)
    rays = torch.from_numpy(dataset['rays_train']).to(device)

    if opt.render_only:
        with torch.no_grad():
            rend_dir = os.path.join(result_dir, 'render_only_{:06d}'.format(global_step))
            os.makedirs(rend_dir, exist_ok=True)
            rgbs, disps = render_path(test_poses[:1], dataset['hwf'], opt.render_chunk, rend_dir, **test_kwargs)
            rgb_dir = os.path.join(rend_dir, 'rgb.mp4')
            disp_dir = os.path.join(rend_dir, 'disp.mp4')
            imageio.mimwrite(rgb_dir, tobyte(rgbs), fps=30, quality=8)
            imageio.mimwrite(disp_dir, tobyte(disps / np.max(disps)), fps=30, quality=8)
            sys.exit()
    i_batch = 0
    for i in trange(global_step + 1, opt.N_iters + 1):
        num = int(rays.shape[0])
        if i_batch >= num:
            i_batch = 0
            idx = torch.randperm(num)
            rays = rays[idx]
            print('shuffle data after an epoch')
        batch = rays[i_batch: i_batch + opt.N_batch_rays]
        batch_rays, target = batch[:, :2], batch[:, -1]
        i_batch += opt.N_batch_rays
        out = render(batch_rays, dataset['hwf'], opt.render_chunk, **train_kwargs)
        rgb = out['rgb']
        rgb0 = out['rgb0']
        optimizer.zero_grad()
        loss = img2mse(rgb, target)
        psnr = mse2psnr(loss)
        loss += img2mse(target, rgb0)
        loss.backward()
        optimizer.step()
        # update learning rate
        decay_rate = 0.1
        lr_decay = opt.lr_decay * 1000
        new_lr = opt.lr * (decay_rate ** (i / lr_decay))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        if i % opt.i_print == 0:
            tqdm.write(f"[Train] Iter: {i} Loss: {loss.item()} PSNR: {psnr.item()}")
        if i % opt.i_weights == 0:
            dir_w = os.path.join(result_dir, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': i,
                'optimizer_dict': optimizer.state_dict(),
                'model_dict': train_kwargs['model'].state_dict(),
                'model_fn_dict': train_kwargs['model_fn'].state_dict()
            }, dir_w)
            print(f'checkpoints saved at {result_dir}')
        if i % opt.i_video == 0:
            test_dir = os.path.join(result_dir, 'spiral_{:06d}'.format(i))
            os.makedirs(test_dir, exist_ok=True)
            with torch.no_grad():
                rgbs, disps = render_path(test_poses, dataset['hwf'], opt.render_chunk, test_dir, **test_kwargs)
            save_dir_rgb = os.path.join(test_dir, 'rgb.mp4')
            save_dir_disp = os.path.join(test_dir, 'disp.mp4')
            imageio.mimwrite(save_dir_rgb, tobyte(rgbs), fps=30, quality=8)
            imageio.mimwrite(save_dir_disp, tobyte(disps/np.max(disps)), fps=30, quality=8)
        if i % opt.i_test == 0:
            val_dir = os.path.join(result_dir, 'val_{:06d}'.format(i))
            os.makedirs(val_dir, exist_ok=True)
            with torch.no_grad():
                render_path(val_poses, dataset['hwf'], opt.render_chunk, val_dir, **test_kwargs)
        # torch.cuda.empty_cache()
    print(f'time for project: {datetime.datetime.now() - begin}')
