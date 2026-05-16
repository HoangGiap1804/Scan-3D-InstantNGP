import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching
from utils import custom_meshgrid
import grid

class NeRFRenderer(nn.Module):
    def __init__(self,
                 bound=1,
                 cuda_ray=False,
                 density_scale=1, # scale up deltas (or sigmas), to make the density grid more sharp. larger value than 1 usually improves performance.
                 min_near=0.2,
                 density_thresh=0.01,
                 bg_radius=-1,
                 ):
        super().__init__()

        self.bound = bound
        self.cascade = max(1, 1 + math.ceil(math.log2(bound)))
        self.grid_size = 128
        self.density_scale = density_scale
        self.min_near = min_near
        self.density_thresh = density_thresh
        self.bg_radius = bg_radius # radius of the background sphere.

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        # extra state for cuda raymarching
        self.cuda_ray = cuda_ray
        if cuda_ray:
            # density grid
            density_grid = torch.zeros([self.cascade, self.grid_size ** 3]) # [CAS, H * H * H]
            density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8) # [CAS * H * H * H // 8]
            self.register_buffer('density_grid', density_grid)
            self.register_buffer('density_bitfield', density_bitfield)
            self.mean_density = 0
            self.iter_density = 0
            # step counter
            step_counter = torch.zeros(16, 2, dtype=torch.int32) # 16 is hardcoded for averaging...
            self.register_buffer('step_counter', step_counter)
            self.mean_count = 0
            self.local_step = 0
    
    def forward(self, x, d):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def density(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def background(self, x, d):
        raise NotImplementedError()

    def reset_extra_state(self):
        if not self.cuda_ray:
            return 
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0



    def run_cuda(self, rays_o, rays_d, dt_gamma=0, bg_color=None, perturb=False, force_all_rays=False, max_steps=1024, T_thresh=1e-4, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            sph = raymarching.sph_from_ray(rays_o, rays_d, self.bg_radius) # [N, 2] in [-1, 1]
            bg_color = self.background(sph, rays_d) # [N, 3]
        elif bg_color is None:
            bg_color = 1

        results = {}

        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 16]
            counter.zero_() # set to 0
            self.local_step += 1

            xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, counter, self.mean_count, perturb, 128, force_all_rays, dt_gamma, max_steps)

            
            sigmas, rgbs = self(xyzs, dirs)
            sigmas = self.density_scale * sigmas

            weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, deltas, rays, T_thresh)
            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth = torch.clamp(depth - nears, min=0) / (fars - nears)
            image = image.view(*prefix, 3)
            depth = depth.view(*prefix)
            
            results['weights_sum'] = weights_sum

        else:
           
            # allocate outputs 
            dtype = torch.float32
            
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            
            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            rays_t = nears.clone() # [N]

            step = 0
            
            while step < max_steps:

                # count alive rays 
                n_alive = rays_alive.shape[0]
                
                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                # Increase n_step to reduce Python overhead (number of while loops)
                n_step = max(min(N // n_alive, 64), 1)

                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, 128, perturb if step == 0 else False, dt_gamma, max_steps)

                sigmas, rgbs = self(xyzs, dirs)
                sigmas = self.density_scale * sigmas

                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image, T_thresh)

                rays_alive = rays_alive[rays_alive >= 0]

                step += n_step

            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth = torch.clamp(depth - nears, min=0) / (fars - nears)
            image = image.view(*prefix, 3)
            depth = depth.view(*prefix)
        
        results['depth'] = depth
        results['image'] = image

        return results

    @torch.no_grad()
    def mark_untrained_grid(self, poses, intrinsic, S=64):
        grid.mark_untrained_grid(self, poses, intrinsic, S)

    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        grid.update_extra_state(self, decay, S)


    def render(self, rays_o, rays_d, staged=False, max_ray_batch=4096, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]

        _run = self.run_cuda

        B, N = rays_o.shape[:2]
        device = rays_o.device

        # never stage when cuda_ray
        if staged and not self.cuda_ray:
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], **kwargs)
                    depth[b:b+1, head:tail] = results_['depth']
                    image[b:b+1, head:tail] = results_['image']
                    head += max_ray_batch
            
            results = {}
            results['depth'] = depth
            results['image'] = image

        else:
            results = _run(rays_o, rays_d, **kwargs)

        return results
