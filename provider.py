import os
import cv2
import json
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import get_rays, srgb_to_linear

def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose

class NeRFDataset:
    def __init__(self, path, type='train', device='cuda', downscale=1, n_test=10, num_rays=4096):
        super().__init__()
        
        self.root_path = path
        self.type = type # train, val, test
        self.downscale = downscale
        self.device = device
        self.training = self.type in ['train', 'all', 'trainval']
        
        # Default options matching torch-ngp
        self.scale = 0.33
        self.offset = [0, 0, 0]
        self.bound = 2
        self.fp16 = True
        self.color_space = 'srgb'
        self.num_rays = num_rays if self.training else -1

        # Load transforms.json
        json_path = os.path.join(self.root_path, f'transforms_{type}.json')
        if not os.path.exists(json_path):
             # Fallback to transforms.json if specific split doesn't exist
             json_path = os.path.join(self.root_path, 'transforms.json')
        
        with open(json_path, 'r') as f:
            transform = json.load(f)

        # Load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            self.H = self.W = None
        
        # Read images and poses
        frames = transform["frames"]
        self.poses = []
        self.images = []

        for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
            f_path = os.path.join(self.root_path, f['file_path'])
            if '.' not in os.path.basename(f_path):
                f_path += '.png'

            if not os.path.exists(f_path):
                continue
            
            pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
            pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

            image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] or [H, W, 4]
            if self.H is None or self.W is None:
                self.H = image.shape[0] // downscale
                self.W = image.shape[1] // downscale

            if image.shape[-1] == 3: 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

            if image.shape[0] != self.H or image.shape[1] != self.W:
                image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                
            # image = image.astype(np.float32) / 255 # [H, W, 3/4]
            image = image.astype(np.uint8) # [H, W, 3/4]

            self.poses.append(pose)
            self.images.append(image)
            
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        
        # Load intrinsics
        if 'camera_angle_x' in transform:
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2))
            fl_y = fl_x
        elif 'fl_x' in transform:
            fl_x = transform['fl_x'] / downscale
            fl_y = transform['fl_y'] / downscale
        else:
            raise RuntimeError('Failed to load focal length!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

    def collate(self, index):
        B = len(index) # list of length 1

        index = index[0]
        poses = self.poses[index:index+1].to(self.device) # [B, 4, 4]
        
        # Patch size is usually 1 unless specified for LPIPS
        rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map=None, patch_size=1)

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
        }

        if self.images is not None:
            images = self.images[index:index+1].to(self.device).float() / 255 # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                # Sample pixels if training
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images
            
        return results

    def dataloader(self):
        size = len(self.poses)
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        return loader
