# Copyright (c) 2023 
# Brandon Y. Feng, University of Maryland, College Park and Rice University. All rights reserved

import os, time, imageio, tqdm, argparse
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from datetime import datetime

import torch
print(f"Using PyTorch Version: {torch.__version__}")
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True #was false set to true
torch.cuda.empty_cache()

import torch.nn.functional as F
from torch.fft import fft2, fftshift
from networks import *
from utils import *
from dataset import *

DEVICE = 'cuda'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='.', type=str)
    parser.add_argument('--data_dir', default='.', type=str)
    parser.add_argument('--viz_dir', default='.', type=str)
    parser.add_argument('--scene_name', default='0609', type=str)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--num_t', default=100, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--width', default=256, type=int)
    parser.add_argument('--vis_freq', default=1000, type=int)
    parser.add_argument('--init_lr', default=1e-3, type=float)
    parser.add_argument('--final_lr', default=1e-3, type=float)
    parser.add_argument('--silence_tqdm', action='store_true')
    parser.add_argument('--save_per_frame', action='store_true')
    parser.add_argument('--static_phase', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--max_intensity', default=0, type=float)
    parser.add_argument('--im_prefix', default='SLM_raw', type=str)
    parser.add_argument('--zero_freq', default=-1, type=int)
    parser.add_argument('--phs_layers', default=2, type=int)
    parser.add_argument('--dynamic_scene', action='store_true')
    parser.add_argument('--use_gsplat', action='store_true', help='Use 2D Gaussian Splatting for image network')
    parser.add_argument('--num_gaussians', default=1000, type=int, help='Number of Gaussians for splatting')
    parser.add_argument('--gs_model_type', default='2dgs', type=str, choices=['2dgs', '3dgs'], help='Gaussian splatting model type')
    parser.add_argument('--gs_init_scale', default=0.05, type=float, help='Initial scale for Gaussians (smaller = sharper)')
    parser.add_argument('--gs_opacity_reg', default=0.01, type=float, help='L1 regularization on opacity (prevent overfitting)')
    parser.add_argument('--gs_scale_reg', default=0.01, type=float, help='Scale regularization (keep Gaussians small)')
    parser.add_argument('--im_lr', default=None, type=float, help='Learning rate for image network (defaults to init_lr)')
    parser.add_argument('--ph_lr', default=None, type=float, help='Learning rate for phase network (defaults to init_lr)')
    parser.add_argument('--grad_clip', default=1.0, type=float, help='Gradient clipping value (0 = no clipping)')
    parser.add_argument('--warmup_epochs', default=10, type=int, help='Number of warmup epochs with lower LR')

    args = parser.parse_args()
    PSF_size = args.width

    ############
    # Setup output folders
    timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    data_dir = f'{args.root_dir}/{args.data_dir}'
    vis_dir = f'{args.viz_dir}/vis/{args.scene_name}_{timestamp}'
    os.makedirs(f'{args.viz_dir}/vis', exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(f'{vis_dir}/final', exist_ok=True)
    print(f'Saving output at: {vis_dir}')
    if args.save_per_frame:
        os.makedirs(f'{vis_dir}/final/per_frame', exist_ok=True)

    ############
    # Training preparations
    dset = BatchDataset(data_dir, num=args.num_t, im_prefix=args.im_prefix, max_intensity=args.max_intensity, zero_freq=args.zero_freq)
    x_batches = torch.cat(dset.xs, axis=0).unsqueeze(1).to(DEVICE)
    y_batches = torch.stack(dset.ys, axis=0).to(DEVICE)

    if args.dynamic_scene:
        net = MovingDiffuse(width=args.width, PSF_size=PSF_size, use_FFT=True, bsize=args.batch_size, phs_layers=args.phs_layers, static_phase=args.static_phase)
    else:
        net = StaticDiffuseNet(width=args.width, PSF_size=PSF_size, use_FFT=True, bsize=args.batch_size, phs_layers=args.phs_layers, static_phase=args.static_phase, use_gsplat=args.use_gsplat, num_gaussians=args.num_gaussians, gs_model_type=args.gs_model_type, gs_init_scale=args.gs_init_scale)

    net = net.to(DEVICE)
    net = torch.compile(net)

    # Separate learning rates for image and phase networks
    im_lr = args.im_lr if args.im_lr is not None else args.init_lr
    ph_lr = args.ph_lr if args.ph_lr is not None else args.init_lr

    im_opt = torch.optim.Adam(net.g_im.parameters(), lr=im_lr)
    ph_opt = torch.optim.Adam(net.g_g.parameters(), lr=ph_lr)
    im_sche = torch.optim.lr_scheduler.CosineAnnealingLR(im_opt, T_max = args.num_epochs, eta_min=args.final_lr)
    ph_sche = torch.optim.lr_scheduler.CosineAnnealingLR(ph_opt, T_max = args.num_epochs, eta_min=args.final_lr)

    print(f'Image LR: {im_lr}, Phase LR: {ph_lr}')

    total_it = 0
    mse_history = []
    t = tqdm.trange(args.num_epochs, disable=args.silence_tqdm)

    ############
    # Training loop
    t0 = time.time()
    for epoch in t:
        # Warmup: gradually increase learning rate in early epochs
        if epoch < args.warmup_epochs:
            warmup_factor = (epoch + 1) / args.warmup_epochs
            for param_group in im_opt.param_groups:
                param_group['lr'] = im_lr * warmup_factor
            for param_group in ph_opt.param_groups:
                param_group['lr'] = ph_lr * warmup_factor

        idxs = torch.randperm(len(dset)).long().to(DEVICE)
        for it in range(0, len(dset), args.batch_size):
            idx = idxs[it:it+args.batch_size]
            x_batch, y_batch = x_batches[idx], y_batches[idx]
            cur_t = (idx / (args.num_t - 1)) - 0.5
            im_opt.zero_grad();  ph_opt.zero_grad()

            y, _kernel, sim_g, sim_phs, I_est = net(x_batch, cur_t)

            mse_loss = F.mse_loss(y, y_batch)

            # Regularization for Gaussian Splatting
            reg_loss = 0.0
            if args.use_gsplat:
                # L1 sparsity on opacities (encourage fewer active Gaussians)
                opacity_reg = args.gs_opacity_reg * torch.mean(torch.sigmoid(net.g_im.opacities))

                # Scale regularization (keep Gaussians small to prevent blur)
                scale_reg = args.gs_scale_reg * torch.mean(net.g_im.scales)

                reg_loss = opacity_reg + scale_reg

            loss = mse_loss + reg_loss
            loss.backward()

            # Gradient clipping to prevent instability
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.g_im.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(net.g_g.parameters(), args.grad_clip)

            ph_opt.step()
            im_opt.step()

            mse_history.append(mse_loss.item())
            if args.use_gsplat:
                t.set_postfix(MSE=f'{mse_loss.item():.4e}', Reg=f'{reg_loss.item():.4e}')
            else:
                t.set_postfix(MSE=f'{mse_loss.item():.4e}')

            if args.vis_freq > 0 and (total_it % args.vis_freq) == 0:
                y, _kernel, sim_g, sim_phs, I_est = net(x_batch, torch.zeros_like(cur_t) - 0.5)

                Abe_est = fftshift(fft2(dset.a_slm.to(DEVICE) * sim_g, norm="forward"), dim=[-2, -1]).abs() ** 2
                if I_est.shape[0] > 1:
                    I_est = I_est[0:1]
                I_est = torch.clamp(I_est, 0, 1)
                yy = F.conv2d(I_est, Abe_est, padding='same').squeeze(0)

                fig, ax = plt.subplots(2, 4, figsize=(32, 16))

                # Row 1: Original visualizations
                ax[0, 0].imshow(y_batch[0].detach().cpu().squeeze(), vmin=0, vmax=1, cmap='gray')
                ax[0, 0].axis('off')
                ax[0, 0].title.set_text('Real Measurement')

                ax[0, 1].imshow(y[0].detach().cpu().squeeze(), vmin=0, vmax=1, cmap='gray')
                ax[0, 1].axis('off')
                ax[0, 1].title.set_text('Sim Measurement')

                ax[0, 2].imshow(I_est.detach().cpu().squeeze(), cmap='gray')
                ax[0, 2].axis('off')
                ax[0, 2].title.set_text('I_est')

                ax[0, 3].imshow(sim_phs[0].detach().cpu().squeeze() % np.pi, cmap='rainbow')
                ax[0, 3].axis('off')
                ax[0, 3].title.set_text(f'Sim Phase Error at t={idx[0]}')

                # Row 2: Kernel, Abe_est, Gaussians (if using gsplat), MSE
                ax[1, 0].imshow(_kernel[0].detach().cpu().squeeze(), cmap='gray')
                ax[1, 0].axis('off')
                ax[1, 0].title.set_text('Sim post-SLM PSF')

                ax[1, 1].imshow(yy[0].squeeze().detach().cpu(), vmin=0, vmax=1, cmap='gray')
                ax[1, 1].axis('off')
                ax[1, 1].title.set_text(f'Abe_est * I_est at t={idx[0]}')

                # Gaussian visualization (if using gsplat)
                if args.use_gsplat:
                    means = net.g_im.means.detach().cpu().numpy()
                    opacities = torch.sigmoid(net.g_im.opacities).detach().cpu().numpy()
                    colors = torch.sigmoid(net.g_im.rgbs).detach().cpu().numpy().squeeze()

                    # Both 2DGS and 3DGS have 3D means, visualize XY projection
                    ax[1, 2].scatter(means[:, 0], means[:, 1], c=colors, s=opacities*50, alpha=0.6, cmap='gray', vmin=0, vmax=1)
                    ax[1, 2].set_xlim(-2, 2)
                    ax[1, 2].set_ylim(-2, 2)
                    ax[1, 2].set_aspect('equal')
                    ax[1, 2].title.set_text(f'Gaussian Positions ({args.gs_model_type.upper()}, N={net.g_im.num_gaussians})')
                    ax[1, 2].set_xlabel('X')
                    ax[1, 2].set_ylabel('Y')
                    ax[1, 2].grid(True, alpha=0.3)
                else:
                    ax[1, 2].axis('off')
                    ax[1, 2].title.set_text('N/A (not using gsplat)')

                # MSE plot
                if len(mse_history) > 0:
                    ax[1, 3].plot(mse_history, linewidth=1)
                    ax[1, 3].set_xlabel('Iteration')
                    ax[1, 3].set_ylabel('MSE Loss')
                    ax[1, 3].set_title('Training MSE')
                    ax[1, 3].grid(True, alpha=0.3)
                    ax[1, 3].set_yscale('log')
                else:
                    ax[1, 3].axis('off')

                plt.tight_layout()
                plt.savefig(f'{vis_dir}/e_{epoch}_it_{it}.jpg')
                plt.close()
                sio.savemat(f'{vis_dir}/Sim_Phase.mat', {'angle': sim_phs.detach().cpu().squeeze().numpy()})

            total_it += 1

        # Only step schedulers after warmup
        if epoch >= args.warmup_epochs:
            im_sche.step()
            ph_sche.step()

    t1 = time.time()
    print(f'Training takes {t1 - t0} seconds.')
    os.makedirs(f'{vis_dir}/final/per_frame', exist_ok=True) 
    ############
    # Export final results
    out_errs = []
    out_abes = []
    out_Iest = []
    for t in range(args.num_t):
        cur_t = (t / (args.num_t - 1)) - 0.5
        cur_t = torch.FloatTensor([cur_t]).to(DEVICE)

        I_est, sim_g, sim_phs = net.get_estimates(cur_t)
        I_est = torch.clamp(I_est, 0, 1).squeeze().detach().cpu().numpy()

        out_Iest.append(I_est)

        est_g = sim_g.detach().cpu().squeeze().numpy()
        out_errs.append(np.uint8(ang_to_unit(np.angle(est_g)) * 255))
        abe = sim_phs[0].detach().cpu().squeeze()
        abe = (abe - abe.min()) / (abe.max() - abe.min())
        out_abes.append(np.uint8(abe * 255))
        if args.save_per_frame and not args.static_phase:
          sio.savemat(f'{vis_dir}/final/per_frame/sim_phase_{t}.mat', {'angle': sim_phs.detach().cpu().squeeze().numpy()})

    if args.dynamic_scene:
        out_Iest = [np.uint8(im * 255) for im in out_Iest]
        imageio.mimsave(f'{vis_dir}/final/final_I.gif', out_Iest, duration=1000*1./30)
    else:
        I_est = np.uint8(I_est.squeeze() * 255)
        imageio.imsave(f'{vis_dir}/final/final_I_est.png', I_est)

    if args.static_phase:
        imageio.imsave(f'{vis_dir}/final/final_aberrations_angle.png', out_errs[0])
        imageio.imsave(f'{vis_dir}/final/final_aberrations.png', out_abes[0])
    else:
        imageio.mimsave(f'{vis_dir}/final/final_aberrations_angle_grey.gif', out_errs, duration=1000*1./30)
        imageio.mimsave(f'{vis_dir}/final/final_aberrations.gif', out_abes, duration=1000*1./30)

    print("Training concludes.")

    colored_err = []
    for i, a in enumerate(out_errs):
        plt.imsave(f'{vis_dir}/final/per_frame/{i:03d}.jpg', a, cmap='rainbow')
        colored_err.append(imageio.imread(f'{vis_dir}/final/per_frame/{i:03d}.jpg'))
    imageio.mimsave(f'{vis_dir}/final/final_aberrations_angle.gif', colored_err, duration=1000*1./30)