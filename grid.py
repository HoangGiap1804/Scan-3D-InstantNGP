import torch
import numpy as np
import raymarching
from utils import custom_meshgrid

@torch.no_grad()
def mark_untrained_grid(renderer, poses, intrinsic, S=64):
    """
    Đánh dấu các vùng trong lưới density mà không có camera nào nhìn thấy là -1.
    """
    if not renderer.cuda_ray:
        return
    
    if isinstance(poses, np.ndarray):
        poses = torch.from_numpy(poses)

    B = poses.shape[0]
    fx, fy, cx, cy = intrinsic
    
    device = renderer.density_bitfield.device
    X = torch.arange(renderer.grid_size, dtype=torch.int32, device=device).split(S)
    Y = torch.arange(renderer.grid_size, dtype=torch.int32, device=device).split(S)
    Z = torch.arange(renderer.grid_size, dtype=torch.int32, device=device).split(S)

    count = torch.zeros_like(renderer.density_grid)
    poses = poses.to(count.device)

    for xs in X:
        for ys in Y:
            for zs in Z:
                # construct points
                xx, yy, zz = custom_meshgrid(xs, ys, zs)
                coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                indices = raymarching.morton3D(coords).long() # [N]
                world_xyzs = (2 * coords.float() / (renderer.grid_size - 1) - 1).unsqueeze(0) # [1, N, 3] in [-1, 1]

                # cascading
                for cas in range(renderer.cascade):
                    bound = min(2 ** cas, renderer.bound)
                    half_grid_size = bound / renderer.grid_size
                    cas_world_xyzs = world_xyzs * (bound - half_grid_size)

                    # split batch to avoid OOM
                    head = 0
                    while head < B:
                        tail = min(head + S, B)
                        # world2cam transform
                        cam_xyzs = cas_world_xyzs - poses[head:tail, :3, 3].unsqueeze(1)
                        cam_xyzs = cam_xyzs @ poses[head:tail, :3, :3] # [S, N, 3]
                        
                        # query if point is covered by any camera
                        mask_z = cam_xyzs[:, :, 2] > 0 # [S, N]
                        mask_x = torch.abs(cam_xyzs[:, :, 0]) < cx / fx * cam_xyzs[:, :, 2] + half_grid_size * 2
                        mask_y = torch.abs(cam_xyzs[:, :, 1]) < cy / fy * cam_xyzs[:, :, 2] + half_grid_size * 2
                        mask = (mask_z & mask_x & mask_y).sum(0).reshape(-1) # [N]

                        # update count 
                        count[cas, indices] += mask
                        head += S
    
    renderer.density_grid[count == 0] = -1
    print(f'[mark untrained grid] {(count == 0).sum()} from {renderer.grid_size ** 3 * renderer.cascade}')

@torch.no_grad()
def update_extra_state(renderer, decay=0.95, S=128):
    """
    Cập nhật density grid và bitfield dựa trên giá trị density thực tế từ model.
    """
    if not renderer.cuda_ray:
        return 
    
    device = renderer.density_bitfield.device
    tmp_grid = - torch.ones_like(renderer.density_grid)
    
    # full update.
    if renderer.iter_density < 16:
        X = torch.arange(renderer.grid_size, dtype=torch.int32, device=device).split(S)
        Y = torch.arange(renderer.grid_size, dtype=torch.int32, device=device).split(S)
        Z = torch.arange(renderer.grid_size, dtype=torch.int32, device=device).split(S)

        for xs in X:
            for ys in Y:
                for zs in Z:
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    indices = raymarching.morton3D(coords).long()
                    xyzs = 2 * coords.float() / (renderer.grid_size - 1) - 1

                    for cas in range(renderer.cascade):
                        bound = min(2 ** cas, renderer.bound)
                        half_grid_size = bound / renderer.grid_size
                        cas_xyzs = xyzs * (bound - half_grid_size)
                        cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                        sigmas = renderer.density(cas_xyzs)['sigma'].reshape(-1).detach().float()
                        sigmas *= renderer.density_scale
                        tmp_grid[cas, indices] = sigmas

    # partial update
    else:
        N = renderer.grid_size ** 3 // 4
        for cas in range(renderer.cascade):
            coords = torch.randint(0, renderer.grid_size, (N, 3), device=device)
            indices = raymarching.morton3D(coords).long()
            occ_indices = torch.nonzero(renderer.density_grid[cas] > 0).squeeze(-1)
            if occ_indices.shape[0] > 0:
                rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long, device=device)
                occ_indices = occ_indices[rand_mask]
                occ_coords = raymarching.morton3D_invert(occ_indices)
                indices = torch.cat([indices, occ_indices], dim=0)
                coords = torch.cat([coords, occ_coords], dim=0)

            xyzs = 2 * coords.float() / (renderer.grid_size - 1) - 1
            bound = min(2 ** cas, renderer.bound)
            half_grid_size = bound / renderer.grid_size
            cas_xyzs = xyzs * (bound - half_grid_size)
            cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
            sigmas = renderer.density(cas_xyzs)['sigma'].reshape(-1).detach().float()
            sigmas *= renderer.density_scale
            tmp_grid[cas, indices] = sigmas

    # ema update
    valid_mask = (renderer.density_grid >= 0) & (tmp_grid >= 0)
    renderer.density_grid[valid_mask] = torch.maximum(renderer.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
    renderer.mean_density = torch.mean(renderer.density_grid.clamp(min=0)).item()
    renderer.iter_density += 1

    # convert to bitfield
    density_thresh = min(renderer.mean_density, renderer.density_thresh)
    renderer.density_bitfield = raymarching.packbits(renderer.density_grid, density_thresh, renderer.density_bitfield)

    ### update step counter
    total_step = min(16, renderer.local_step)
    if total_step > 0:
        renderer.mean_count = int(renderer.step_counter[:total_step, 0].sum().item() / total_step)
    renderer.local_step = 0
