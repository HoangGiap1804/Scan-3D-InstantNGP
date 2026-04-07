import torch
import cv2
import numpy as np
from packaging import version as pver

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)

@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None, patch_size=1):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device)) # float
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        # if use patch-based sampling, ignore error_map
        if patch_size > 1:

            # random sample left-top cores.
            num_patch = N // (patch_size ** 2)
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1) # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1) # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0) # [np, p^2, 2]
            inds = inds.view(-1, 2) # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1] # [N], flatten

            inds = inds.expand([B, N])

        elif error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results

def seed_everything(seed):
    import random
    import os
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

@torch.no_grad()
def render_full_image(model, pose, intrinsics, H, W, bg_color=0.0, return_float=False, **kwargs):
    ''' render a full image from a pose
    Args:
        model: NeRFRenderer
        pose: [1, 4, 4]
        intrinsics: [fl_x, fl_y, cx, cy]
        H, W: int
        bg_color: float, list of 3 floats, or torch.Tensor
        return_float: if True, returns float32 numpy array [0, 1], else uint8.
        **kwargs: additional arguments for model.render
    Returns:
        image: [H, W, 3], uint8 or float32
    '''
    device = pose.device
    rays = get_rays(pose, intrinsics, H, W)
    rays_o = rays['rays_o']
    rays_d = rays['rays_d']
    
    # staged rendering for full image to avoid OOM
    with torch.cuda.amp.autocast(enabled=True):
        outputs = model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, **kwargs)
        image = outputs['image'].reshape(H, W, 3)
        
        # convert linear to srgb
        image = linear_to_srgb(image)
        
    if return_float:
        return image.cpu().numpy()
    
    return (image.cpu().numpy() * 255).astype(np.uint8)


def save_video(images, path, fps=10):
    ''' save a list of images to a video using OpenCV
    Args:
        images: list of [H, W, 3] uint8
        path: str
        fps: int
    '''
    if len(images) == 0:
        return
    
    # Force .avi extension for better compatibility with XVID
    if not path.endswith('.avi'):
        path = os.path.splitext(path)[0] + '.avi'
    
    H, W, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, fps, (W, H))
    
    for img in images:
        # OpenCV uses BGR, so we need to convert from RGB
        out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
    out.release()
    print(f"Video saved to {path}")
def get_orbit_pose(azimuth, elevation, radius, center=np.array([0, 0, 0], dtype=np.float32)):
    ''' get a camera pose for orbit viewing
    Args:
        azimuth, elevation: float (degrees)
        radius: float
        center: [3]
    Returns:
        pose: [4, 4]
    '''
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)

    # Position in Cartesian coordinates
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    pos = np.array([x, y, z], dtype=np.float32) + center

    # Forward direction (towards center, OpenCV +Z points into screen)
    forward = center - pos
    forward /= np.linalg.norm(forward)

    # World Up direction (assume Y is up)
    world_up = np.array([0, 1, 0], dtype=np.float32)
    
    # Right direction (OpenCV +X)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)

    # Down direction to match OpenCV camera (OpenCV +Y points down)
    down = np.cross(forward, right)

    # Camera to World matrix
    # [R | T]
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = down
    pose[:3, 2] = forward
    pose[:3, 3] = pos

    return pose
