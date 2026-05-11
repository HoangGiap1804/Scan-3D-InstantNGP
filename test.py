import torch
import numpy as np
import os
import math
import cv2
import tqdm

from model import NeRFNetwork
from provider import NeRFDataset
from utils import seed_everything, render_full_image, save_video

# ---------------------------------------------------------------------------
# Simple PSNR meter
# ---------------------------------------------------------------------------
class PSNRMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.V = 0.0
        self.N = 0

    def update(self, pred, gt, mask=None):
        """pred, gt: torch tensor [*, 3], values in [0, 1]"""
        with torch.no_grad():
            if mask is not None:
                # Only calculate MSE on masked pixels
                if mask.any():
                    mse = torch.mean((pred[mask] - gt[mask]) ** 2).item()
                else:
                    mse = 0.0 # Or skip
            else:
                mse = torch.mean((pred - gt) ** 2).item()
            
            if mse > 0:
                psnr = -10.0 * math.log10(max(mse, 1e-10))
            else:
                psnr = 100.0 # Perfect match or no pixels
        
        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / max(self.N, 1)

    def report(self):
        return f"PSNR = {self.measure():.2f} dB"


def test(args):
    # Configuration
    path = args.path
    workspace = args.workspace
    os.makedirs(workspace, exist_ok=True)
    test_dir = os.path.join(workspace, 'results')
    os.makedirs(test_dir, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 1. Load Dataset
    # -------------------------------------------------------------------------
    print(f"Loading test dataset from {path}...")
    test_dataset = NeRFDataset(path, type='test', device=device, downscale=args.downscale)
    
    # Print dataset memory usage
    mem_images = test_dataset.images.element_size() * test_dataset.images.nelement()
    mem_poses = test_dataset.poses.element_size() * test_dataset.poses.nelement()
    total_mem = (mem_images + mem_poses) / (1024 * 1024) # MB
    print(f"Loaded {len(test_dataset.poses)} test frames. Dataset memory usage: {total_mem:.2f} MB")

    # -------------------------------------------------------------------------
    # 2. Initialize Model
    # -------------------------------------------------------------------------
    print(f"Initializing model with bound {args.bound}...")
    model = NeRFNetwork(bound=args.bound, cuda_ray=True).to(device)

    # -------------------------------------------------------------------------
    # 3. Load Checkpoint
    # -------------------------------------------------------------------------
    ckpt_path = args.ckpt if args.ckpt else os.path.join(workspace, "model_epoch_latest.pth")
    
    if not os.path.exists(ckpt_path):
        # Try to find the latest checkpoint in workspace
        ckpts = [f for f in os.listdir(workspace) if f.endswith('.pth')]
        if ckpts:
            def get_epoch(name):
                try:
                    return int(name.split('_')[-1].split('.')[0])
                except ValueError:
                    return 999999
            ckpts.sort(key=get_epoch)
            ckpt_path = os.path.join(workspace, ckpts[-1])

    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)

        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            # Check if we should use EMA
            if args.use_ema and 'ema' in checkpoint:
                print("Using EMA weights...")
                model.load_state_dict(checkpoint['ema'])
            else:
                model.load_state_dict(checkpoint['model'])
        else:
            # Old format: just state_dict
            model.load_state_dict(checkpoint)
    else:
        print(f"Error: No checkpoint found at {ckpt_path}")
        return

    # -------------------------------------------------------------------------
    # 4. Testing Loop
    # -------------------------------------------------------------------------
    model.eval()
    psnr_meter = PSNRMeter()
    
    images_to_save = []
    
    print("Starting testing...")
    if args.ignore_transparent:
        print("Ignoring transparent regions for PSNR calculation.")

    with torch.no_grad():
        for i in tqdm.trange(len(test_dataset.poses)):
            pose = test_dataset.poses[i:i+1].to(device)
            gt_image = test_dataset.images[i].to(device) # [H, W, 3/4]
            
            # Mask for transparent pixels
            mask = None
            if args.ignore_transparent and gt_image.shape[-1] == 4:
                mask = gt_image[..., 3] > 0
            
            # Render image (returns float [H, W, 3])
            pred_image_float = render_full_image(
                model, pose, test_dataset.intrinsics,
                test_dataset.H, test_dataset.W,
                bg_color=0.0, max_steps=args.max_steps,
                return_float=True
            )
            pred_image_float = torch.from_numpy(pred_image_float).to(device)

            # Prepare GT for PSNR (handle alpha channel if present)
            if gt_image.shape[-1] == 4:
                # For visualization, we keep it over black
                gt_rgb = gt_image[..., :3] * gt_image[..., 3:]
            else:
                gt_rgb = gt_image
            
            # Update PSNR
            psnr_meter.update(pred_image_float, gt_rgb, mask=mask)

            # Save image (uint8)
            pred_image_uint8 = (pred_image_float.cpu().numpy() * 255).astype(np.uint8)
            images_to_save.append(pred_image_uint8)
            
            img_path = os.path.join(test_dir, f"{i:04d}.png")
            cv2.imwrite(img_path, cv2.cvtColor(pred_image_uint8, cv2.COLOR_RGB2BGR))

    # -------------------------------------------------------------------------
    # 5. Final Report
    # -------------------------------------------------------------------------
    print("-" * 30)
    print(f"Test finished!")
    print(psnr_meter.report())
    print("-" * 30)

    # Save video
    if args.save_video:
        video_path = os.path.join(workspace, 'test_video.mp4')
        print(f"Generating test video to {video_path}...")
        save_video(images_to_save, video_path, fps=args.fps)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./data', help="Path to data")
    parser.add_argument('--workspace', type=str, default='workspace', help="Workspace directory")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--bound', type=float, default=0.5, help="Scene bound")
    parser.add_argument('--max_steps', type=int, default=1024, help="Max steps per ray")
    parser.add_argument('--ckpt', type=str, default=None, help="Specific checkpoint path to load")
    parser.add_argument('--use_ema', action='store_true', help="Use EMA weights if available")
    parser.add_argument('--save_video', action='store_true', help="Save test images as video")
    parser.add_argument('--fps', type=int, default=30, help="FPS for video")
    parser.add_argument('--downscale', type=int, default=1, help="Downscale images before loading")
    parser.add_argument('--ignore_transparent', action='store_true', help="Ignore transparent pixels in GT for PSNR")

    args = parser.parse_args()
    test(args)
