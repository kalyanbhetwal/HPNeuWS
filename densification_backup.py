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

