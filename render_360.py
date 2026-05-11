import torch
import numpy as np
import os
import cv2
import tqdm
import json
from model import NeRFNetwork
from utils import render_full_image, get_orbit_pose, save_video

def render_360(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Initialize Model
    print(f"Initializing model with bound {args.bound}...")
    model = NeRFNetwork(bound=args.bound, cuda_ray=True).to(device).eval()

    # 2. Load Checkpoint
    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_path = os.path.join(args.workspace, "model.pth")
        if not os.path.exists(ckpt_path):
            ckpts = [f for f in os.listdir(args.workspace) if f.endswith('.pth')]
            if ckpts:
                def get_epoch(name):
                    try:
                        return int(name.split('_')[-1].split('.')[0])
                    except ValueError:
                        return 999999
                ckpts.sort(key=get_epoch)
                ckpt_path = os.path.join(args.workspace, ckpts[-1])

    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Error: No checkpoint found in {args.workspace}")
        return

    # 3. Setup Intrinsics
    H, W = args.res, args.res
    fl = W # Default FOV
    intrinsics = np.array([fl, fl, W / 2, H / 2])

    # Try to load real intrinsics from workspace
    json_paths = [
        os.path.join(args.workspace, 'transforms_train.json'),
        os.path.join(args.workspace, 'transforms.json'),
    ]
    for path in json_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    transform = json.load(f)
                if 'camera_angle_x' in transform:
                    fl_x = W / (2 * np.tan(transform['camera_angle_x'] / 2))
                    fl_y = fl_x
                elif 'fl_x' in transform:
                    scale = W / transform['w'] if 'w' in transform else 1.0
                    fl_x = transform['fl_x'] * scale
                    fl_y = transform['fl_y'] * scale
                else:
                    continue
                
                cx = (transform['cx'] * (W / transform['w'])) if 'cx' in transform else (W / 2)
                cy = (transform['cy'] * (H / transform['h'])) if 'cy' in transform else (H / 2)
                intrinsics = np.array([fl_x, fl_y, cx, cy])
                print(f"Loaded intrinsics from {path}")
                break
            except Exception as e:
                print(f"Failed to load intrinsics from {path}: {e}")

    # 4. Rendering Loop
    frames = []
    print(f"Rendering 360 orbit video ({args.num_frames} frames)...")
    
    with torch.no_grad():
        for i in tqdm.trange(args.num_frames):
            azimuth = i / args.num_frames * 360
            elevation = args.elevation
            radius = args.radius
            
            pose = get_orbit_pose(azimuth, elevation, radius)
            pose = torch.from_numpy(pose).unsqueeze(0).to(device)
            
            image = render_full_image(
                model, pose, intrinsics, H, W, 
                bg_color=args.bg_color,
                return_float=False,
                num_steps=args.num_steps,
                upsample_steps=args.upsample_steps
            )
            frames.append(image)

    # 5. Save Video
    os.makedirs('videos', exist_ok=True)
    save_path = os.path.join('videos', f"{os.path.basename(args.workspace.strip('/'))}_360.mp4")
    save_video(frames, save_path, fps=args.fps)
    print(f"Done! Video saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, default='workspace', help="Path to workspace containing model.pth")
    parser.add_argument('--ckpt', type=str, default=None, help="Specific checkpoint path")
    parser.add_argument('--res', type=int, default=800, help="Video resolution")
    parser.add_argument('--num_frames', type=int, default=120, help="Number of frames for 360 degree")
    parser.add_argument('--fps', type=int, default=30, help="Frames per second")
    parser.add_argument('--radius', type=float, default=4.0, help="Orbit radius")
    parser.add_argument('--elevation', type=float, default=0.0, help="Orbit elevation in degrees")
    parser.add_argument('--bg_color', type=float, default=0.0, help="Background color (0-1)")
    parser.add_argument('--bound', type=float, default=0.5, help="Scene bound")
    parser.add_argument('--num_steps', type=int, default=128, help="Number of steps per ray")
    parser.add_argument('--upsample_steps', type=int, default=128, help="Number of upsample steps per ray")

    args = parser.parse_args()
    render_360(args)
