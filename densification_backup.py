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
    parser.add_argument('--gs_init_radius', default=1.5, type=float, help='Initial radius for circular Gaussian distribution (matches scene extent)')
    parser.add_argument('--uniform_init', action='store_true', help='Initialize Gaussians uniformly across entire image (instead of circular)')
    parser.add_argument('--gs_opacity_reg', default=0.01, type=float, help='L1 regularization on opacity (prevent overfitting)')
    parser.add_argument('--gs_scale_reg', default=0.01, type=float, help='Scale regularization (keep Gaussians small)')
    parser.add_argument('--im_lr', default=None, type=float, help='Learning rate for image network (defaults to init_lr)')
    parser.add_argument('--ph_lr', default=None, type=float, help='Learning rate for phase network (defaults to init_lr)')
    parser.add_argument('--grad_clip', default=1.0, type=float, help='Gradient clipping value (0 = no clipping)')
    parser.add_argument('--warmup_epochs', default=100, type=int, help='Number of warmup epochs with lower LR')
    parser.add_argument('--use_circular_mask', action='store_true', help='Only compute loss in circular region where Gaussians are initialized')
    parser.add_argument('--constrain_gaussians', action='store_true', help='Constrain Gaussian positions to stay inside circular mask')
    parser.add_argument('--position_penalty', default=0.1, type=float, help='Weight for soft position penalty (encourages Gaussians to stay inside circle)')
    parser.add_argument('--densify', action='store_true', help='Densify Gaussians in high-gradient regions (split Gaussians with high gradients)')
    parser.add_argument('--densify_interval', default=100, type=int, help='Densify every N iterations')
    parser.add_argument('--densify_from_iter', default=500, type=int, help='Start densification after N iterations (wait for initial convergence)')
    parser.add_argument('--densify_until_iter', default=15000, type=int, help='Stop densification after N iterations')
    parser.add_argument('--densify_grad_percentile', default=90, type=float, help='Densify Gaussians in top N percentile of accumulated gradients')
    parser.add_argument('--densify_radius', default=None, type=float, help='Only densify Gaussians within this radius from center (None = no restriction)')
    parser.add_argument('--max_gaussians', default=50000, type=int, help='Maximum number of Gaussians after densification')

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

    # Compute and save average of all input measurements
    y_avg = y_batches.mean(dim=0).detach().cpu().numpy()
    os.makedirs(f'{vis_dir}/analysis', exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(y_avg, cmap='gray', vmin=0, vmax=1)
    plt.title(f'Average of {args.num_t} Input Measurements')
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{vis_dir}/analysis/input_average.png', dpi=150, bbox_inches='tight')
    plt.close()
    imageio.imsave(f'{vis_dir}/analysis/input_average_raw.png', np.uint8(y_avg * 255))
    print(f'Saved average input image to {vis_dir}/analysis/input_average.png')

    if args.dynamic_scene:
        net = MovingDiffuse(width=args.width, PSF_size=PSF_size, use_FFT=True, bsize=args.batch_size, phs_layers=args.phs_layers, static_phase=args.static_phase)
    else:
        net = StaticDiffuseNet(width=args.width, PSF_size=PSF_size, use_FFT=True, bsize=args.batch_size, phs_layers=args.phs_layers, static_phase=args.static_phase, use_gsplat=args.use_gsplat, num_gaussians=args.num_gaussians, gs_model_type=args.gs_model_type, gs_init_scale=args.gs_init_scale, init_radius=args.gs_init_radius, uniform_init=args.uniform_init)

    net = net.to(DEVICE)

    # torch.compile incompatible with gsplat custom CUDA kernels
    if not args.use_gsplat:
        net = torch.compile(net)
        print("Using torch.compile for speedup")
    else:
        print("Skipping torch.compile (incompatible with gsplat)")

    # Save circular mask visualization if using masked loss
    if args.use_gsplat and args.use_circular_mask:
        mask = net.g_im.get_loss_mask().cpu().numpy()
        plt.figure(figsize=(8, 8))
        plt.imshow(mask, cmap='gray')
        plt.title(f'Loss Mask (radius={args.gs_init_radius}, area={mask.sum():.0f}/{mask.size} = {100*mask.mean():.1f}%)')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'{vis_dir}/analysis/loss_mask.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Using circular loss mask with radius={args.gs_init_radius}')

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

    # For densification: accumulate gradient statistics over time
    if args.use_gsplat and args.densify:
        grad_accum = torch.zeros(net.g_im.num_gaussians, device=DEVICE)
        grad_count = torch.zeros(net.g_im.num_gaussians, device=DEVICE)

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

            # Apply circular mask in loss computation if enabled
            if args.use_gsplat and args.use_circular_mask:
                mask = net.g_im.get_loss_mask()
                # Expand mask to match batch dimensions
                mask_expanded = mask.unsqueeze(0).expand(y.shape[0], -1, -1)
                # Masked MSE loss
                mse_loss = F.mse_loss(y * mask_expanded, y_batch * mask_expanded)
                # Normalize by mask area to get correct scale
                mask_area = mask.sum()
                total_area = mask.numel()
                mse_loss = mse_loss * (total_area / mask_area)
            else:
                mse_loss = F.mse_loss(y, y_batch)

            # Regularization for Gaussian Splatting
            reg_loss = 0.0
            # if args.use_gsplat:
            #     # L1 sparsity on opacities (encourage fewer active Gaussians)
            #     opacity_reg = args.gs_opacity_reg * torch.mean(torch.sigmoid(net.g_im.opacities))

            #     # Scale regularization (keep Gaussians small to prevent blur)
            #     scale_reg = args.gs_scale_reg * torch.mean(net.g_im.scales)

            #     reg_loss = opacity_reg + scale_reg

            # Soft position penalty (encourages Gaussians to stay inside circle)
            if args.use_gsplat and args.constrain_gaussians:
                # Get XY positions
                positions_xy = net.g_im.means[:, :2]  # [N, 2]

                # Compute distance from center
                dist = torch.sqrt((positions_xy ** 2).sum(dim=1))  # [N]

                # Penalty for Gaussians outside the radius (quadratic penalty)
                # Only penalize those outside, no penalty inside
                violations = torch.relu(dist - args.gs_init_radius)  # 0 if inside, positive if outside
                position_penalty = args.position_penalty * torch.mean(violations ** 2)
                reg_loss = reg_loss + position_penalty

            loss = mse_loss + reg_loss
            loss.backward()

            # Compute gradient magnitude per Gaussian for visualization and densification
            if args.use_gsplat:
                with torch.no_grad():
                    # Gradient magnitude from position gradients
                    grad_mag_tensor = net.g_im.means.grad.norm(dim=1)
                    grad_mag = grad_mag_tensor.detach().cpu().numpy()

                    # Store for visualization (will be used in next vis iteration)
                    if not hasattr(net.g_im, 'grad_magnitudes'):
                        net.g_im.grad_magnitudes = grad_mag
                    else:
                        net.g_im.grad_magnitudes = grad_mag

                    # Accumulate gradients for densification
                    if args.densify:
                        # Resize accumulators if Gaussians were added
                        if grad_accum.shape[0] < net.g_im.num_gaussians:
                            new_size = net.g_im.num_gaussians - grad_accum.shape[0]
                            grad_accum = torch.cat([grad_accum, torch.zeros(new_size, device=DEVICE)])
                            grad_count = torch.cat([grad_count, torch.zeros(new_size, device=DEVICE)])

                        grad_accum += grad_mag_tensor
                        grad_count += 1

            # Gradient clipping to prevent instability
            # if args.grad_clip > 0:
            #     torch.nn.utils.clip_grad_norm_(net.g_im.parameters(), args.grad_clip)
            #     torch.nn.utils.clip_grad_norm_(net.g_g.parameters(), args.grad_clip)

            ph_opt.step()
            im_opt.step()

            # Densification: split high-gradient Gaussians based on accumulated statistics
            if (args.use_gsplat and args.densify and
                total_it % args.densify_interval == 0 and
                total_it >= args.densify_from_iter and
                total_it <= args.densify_until_iter):

                with torch.no_grad():
                    # Compute average gradient magnitude per Gaussian
                    avg_grad = grad_accum / (grad_count + 1e-8)

                    # Use percentile-based threshold (adaptive to current distribution)
                    threshold = np.percentile(avg_grad.cpu().numpy(), args.densify_grad_percentile)

                    # Find Gaussians with consistently high gradients
                    high_grad_mask = avg_grad > threshold

                    # Filter by position if densify_radius is set (only densify near center)
                    if args.densify_radius is not None:
                        positions_xy = net.g_im.means[:, :2]
                        dist_from_center = torch.sqrt((positions_xy ** 2).sum(dim=1))
                        inside_radius_mask = dist_from_center <= args.densify_radius
                        high_grad_mask = high_grad_mask & inside_radius_mask

                    num_to_split = high_grad_mask.sum().item()

                    if num_to_split > 0 and net.g_im.num_gaussians + num_to_split <= args.max_gaussians:
                        # Get indices of high-gradient Gaussians
                        split_indices = torch.where(high_grad_mask)[0]

                        # Clone and perturb these Gaussians
                        new_means = net.g_im.means[split_indices].clone()
                        # Add small random offset to avoid exact duplicates
                        new_means[:, :2] += (torch.randn_like(new_means[:, :2]) * 0.01)

                        new_scales = net.g_im.scales[split_indices].clone() * 0.8  # Slightly smaller
                        new_quats = net.g_im.quats[split_indices].clone()
                        new_opacities = net.g_im.opacities[split_indices].clone()
                        new_rgbs = net.g_im.rgbs[split_indices].clone()

                        # Concatenate new Gaussians
                        net.g_im.means = nn.Parameter(torch.cat([net.g_im.means, new_means], dim=0))
                        net.g_im.scales = nn.Parameter(torch.cat([net.g_im.scales, new_scales], dim=0))
                        net.g_im.quats = nn.Parameter(torch.cat([net.g_im.quats, new_quats], dim=0))
                        net.g_im.opacities = nn.Parameter(torch.cat([net.g_im.opacities, new_opacities], dim=0))
                        net.g_im.rgbs = nn.Parameter(torch.cat([net.g_im.rgbs, new_rgbs], dim=0))

                        # Update num_gaussians
                        net.g_im.num_gaussians += num_to_split

                        # Re-initialize optimizer with new parameters
                        im_opt = torch.optim.Adam(net.g_im.parameters(), lr=im_lr)

                        print(f"\nDensified at iter {total_it}: split {num_to_split} Gaussians (threshold={threshold:.2e}, total={net.g_im.num_gaussians})")

                        # Update grad_magnitudes for visualization (extend with zeros for new Gaussians)
                        if hasattr(net.g_im, 'grad_magnitudes'):
                            net.g_im.grad_magnitudes = np.concatenate([
                                net.g_im.grad_magnitudes,
                                np.zeros(num_to_split)
                            ])

                        # Reset gradient accumulation for next window
                        grad_accum.zero_()
                        grad_count.zero_()

            mse_history.append(mse_loss.item())
            if args.use_gsplat:
                t.set_postfix(MSE=f'{mse_loss.item():.4e}')#, Reg=f'{reg_loss.item():.4e}')
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

                    # Count active Gaussians (opacity > threshold)
                    opacity_threshold = 0.1
                    num_active = (opacities > opacity_threshold).sum()
                    num_total = net.g_im.num_gaussians

                    # Color by gradient magnitude if available, otherwise by brightness
                    if hasattr(net.g_im, 'grad_magnitudes'):
                        grad_mag = net.g_im.grad_magnitudes

                        # Bin into 4 categories based on percentiles
                        p25 = np.percentile(grad_mag, 25)
                        p50 = np.percentile(grad_mag, 50)
                        p75 = np.percentile(grad_mag, 75)

                        # Assign discrete colors: blue (very low), green (low), yellow (medium), red (high)
                        colors = np.zeros((len(grad_mag), 3))
                        colors[grad_mag <= p25] = [0.2, 0.4, 0.8]      # Blue: very low gradient
                        colors[(grad_mag > p25) & (grad_mag <= p50)] = [0.3, 0.8, 0.3]  # Green: low gradient
                        colors[(grad_mag > p50) & (grad_mag <= p75)] = [0.95, 0.8, 0.2] # Yellow: medium gradient
                        colors[grad_mag > p75] = [0.9, 0.2, 0.2]       # Red: high gradient

                        # Create legend handles for gradient quartiles
                        from matplotlib.patches import Patch
                        legend_elements = [
                            Patch(facecolor=[0.2, 0.4, 0.8], edgecolor='black', label='0-25%'),
                            Patch(facecolor=[0.3, 0.8, 0.3], edgecolor='black', label='25-50%'),
                            Patch(facecolor=[0.95, 0.8, 0.2], edgecolor='black', label='50-75%'),
                            Patch(facecolor=[0.9, 0.2, 0.2], edgecolor='black', label='75-100%')
                        ]
                        use_legend = True
                        title_text = f'Gaussians ({num_active}/{num_total} active)'
                    else:
                        colors = torch.sigmoid(net.g_im.rgbs).detach().cpu().numpy().squeeze()
                        colors = np.repeat(colors[:, np.newaxis], 3, axis=1)  # Grayscale to RGB
                        use_legend = False
                        title_text = f'Gaussians ({num_active}/{num_total} active)'

                    # Both 2DGS and 3DGS have 3D means, visualize XY projection
                    scatter = ax[1, 2].scatter(means[:, 0], means[:, 1], c=colors, s=opacities*50, alpha=0.7, edgecolors='black', linewidths=0.3)
                    ax[1, 2].set_xlim(-2, 2)
                    ax[1, 2].set_ylim(-2, 2)
                    ax[1, 2].set_aspect('equal')
                    ax[1, 2].set_title(title_text, fontsize=8)
                    ax[1, 2].set_xlabel('X')
                    ax[1, 2].set_ylabel('Y')
                    ax[1, 2].grid(True, alpha=0.3)
                    if use_legend:
                        ax[1, 2].legend(handles=legend_elements, loc='upper right', fontsize=6, framealpha=0.9)
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
    # Export average reconstruction
    with torch.no_grad():
        all_reconstructions = []
        for t_idx in range(args.num_t):
            cur_t = (t_idx / (args.num_t - 1)) - 0.5
            cur_t = torch.FloatTensor([cur_t]).to(DEVICE)
            I_est, _, _ = net.get_estimates(cur_t)
            I_est = torch.clamp(I_est, 0, 1).squeeze().detach().cpu().numpy()
            all_reconstructions.append(I_est)

        recon_avg = np.mean(all_reconstructions, axis=0)
        plt.figure(figsize=(8, 8))
        plt.imshow(recon_avg, cmap='gray')
        plt.title(f'Average Reconstruction from {args.num_t} Frames')
        plt.colorbar()
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{vis_dir}/analysis/reconstruction_average.png', dpi=150, bbox_inches='tight')
        plt.close()
        imageio.imsave(f'{vis_dir}/analysis/reconstruction_average_raw.png', np.uint8(recon_avg * 255))
        print(f'Saved average reconstruction to {vis_dir}/analysis/reconstruction_average.png')

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

    # Save final Gaussian gradient visualization
    if args.use_gsplat and hasattr(net.g_im, 'grad_magnitudes'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        means = net.g_im.means.detach().cpu().numpy()
        opacities = torch.sigmoid(net.g_im.opacities).detach().cpu().numpy()
        grad_mag = net.g_im.grad_magnitudes

        # Count active Gaussians
        opacity_threshold = 0.1
        num_active = (opacities > opacity_threshold).sum()
        num_total = net.g_im.num_gaussians

        # Bin into 4 categories based on percentiles
        p25 = np.percentile(grad_mag, 25)
        p50 = np.percentile(grad_mag, 50)
        p75 = np.percentile(grad_mag, 75)

        # Left: colored by gradient magnitude (4 discrete colors)
        colors = np.zeros((len(grad_mag), 3))
        colors[grad_mag <= p25] = [0.2, 0.4, 0.8]      # Blue: very low gradient
        colors[(grad_mag > p25) & (grad_mag <= p50)] = [0.3, 0.8, 0.3]  # Green: low gradient
        colors[(grad_mag > p50) & (grad_mag <= p75)] = [0.95, 0.8, 0.2] # Yellow: medium gradient
        colors[grad_mag > p75] = [0.9, 0.2, 0.2]       # Red: high gradient

        sc1 = ax1.scatter(means[:, 0], means[:, 1], c=colors, s=opacities*50, alpha=0.7, edgecolors='black', linewidths=0.5)
        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-2, 2)
        ax1.set_aspect('equal')
        ax1.set_title(f'Gaussians by gradient ({num_active}/{num_total} active)', fontsize=11)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.grid(True, alpha=0.3)

        # Add legend for gradient quartiles
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=[0.2, 0.4, 0.8], edgecolor='black', label='0-25%'),
            Patch(facecolor=[0.3, 0.8, 0.3], edgecolor='black', label='25-50%'),
            Patch(facecolor=[0.95, 0.8, 0.2], edgecolor='black', label='50-75%'),
            Patch(facecolor=[0.9, 0.2, 0.2], edgecolor='black', label='75-100%')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95, title='Gradient percentile')

        # Right: histogram with colored bins
        ax2.hist(grad_mag, bins=50, alpha=0.7, edgecolor='black', color='gray')
        ax2.axvline(p25, color=[0.2, 0.4, 0.8], linestyle='--', linewidth=2, label=f'🔵 25%: {p25:.2e}')
        ax2.axvline(p50, color=[0.3, 0.8, 0.3], linestyle='--', linewidth=2, label=f'🟢 50%: {p50:.2e}')
        ax2.axvline(p75, color=[0.95, 0.8, 0.2], linestyle='--', linewidth=2, label=f'🟡 75%: {p75:.2e}')
        ax2.set_xlabel('Gradient magnitude')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Gradient distribution\n(min={grad_mag.min():.2e}, max={grad_mag.max():.2e})')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{vis_dir}/final/gaussian_gradients.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved Gaussian gradient visualization to {vis_dir}/final/gaussian_gradients.png")

    print("Training concludes.")

    colored_err = []
    for i, a in enumerate(out_errs):
        plt.imsave(f'{vis_dir}/final/per_frame/{i:03d}.jpg', a, cmap='rainbow')
        colored_err.append(imageio.imread(f'{vis_dir}/final/per_frame/{i:03d}.jpg'))
    imageio.mimsave(f'{vis_dir}/final/final_aberrations_angle.gif', colored_err, duration=1000*1./30)