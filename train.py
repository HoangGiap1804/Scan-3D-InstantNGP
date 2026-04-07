import torch
import torch.optim as optim
import tqdm
import numpy as np
import os

from model import NeRFNetwork
from provider import NeRFDataset
from utils import seed_everything, render_full_image, save_video
import cv2

def train(args):
    # Configuration
    path = args.path
    workspace = args.workspace
    os.makedirs(workspace, exist_ok=True)
    
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Dataset
    print(f"Loading dataset from {path}...")
    train_dataset = NeRFDataset(path, type='train', device=device, num_rays=args.num_rays)
    train_loader = train_dataset.dataloader()
    print(f"Loaded {len(train_dataset.poses)} frames.")

    # 2. Initialize Model
    print(f"Initializing model with bound {args.bound}...")
    model = NeRFNetwork(bound=args.bound, cuda_ray=True).to(device)
    print(model)

    # 2.5 Load Checkpoint
    ckpt_path = args.ckpt if args.ckpt else os.path.join(workspace, "model_epoch_200.pth") 
    start_epoch = 0
    
    # Find the latest epoch checkpoint if model.pth doesn't exist
    if not os.path.exists(ckpt_path):
        ckpts = [f for f in os.listdir(workspace) if f.endswith('.pth')]
        if ckpts:
            # Sort by epoch number: model_epoch_20.pth -> 20
            ckpts.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if 'epoch' in x else 0)
            ckpt_path = os.path.join(workspace, ckpts[-1])
    
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Handle both old format (only state_dict) and new format (dict with model, optimizer, epoch)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            # We will load optimizer state after initializing it
            start_epoch = checkpoint['epoch']
        else:
            model.load_state_dict(checkpoint)
            # Try to guess start_epoch from filename
            if 'epoch' in ckpt_path:
                start_epoch = int(ckpt_path.split('_')[-1].split('.')[0])
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("No checkpoint found, starting from scratch.")

    print("Starting training...")
    epochs = args.epochs
    global_step = start_epoch * len(train_loader)

    # 3. Optimizer & Scheduler
    optimizer = optim.Adam(model.get_params(lr=args.lr), betas=(0.9, 0.99), eps=1e-8)
    
    # Initialize AMP GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    
    # Continuous decay scheduler
    max_steps = epochs * len(train_loader)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** (iter / max_steps))

    # Load optimizer state if available
    if os.path.exists(ckpt_path) and isinstance(checkpoint, dict) and 'optimizer' in checkpoint:
        print("Loading optimizer state...")
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
    
    # Fast forward scheduler
    for _ in range(global_step):
        scheduler.step()

    # 4. Training Loop
    model.train()
    criterion = torch.nn.MSELoss(reduction='none')

    # Validation setup
    val_dir = os.path.join(workspace, 'validation')
    os.makedirs(val_dir, exist_ok=True)
    images = []
    val_pose = train_dataset.poses[0:1].to(device)
    val_intrinsics = train_dataset.intrinsics
    val_H, val_W = train_dataset.H, train_dataset.W

    # mark untrained region
    if model.cuda_ray:
        model.mark_untrained_grid(train_dataset.poses, train_dataset.intrinsics)

    for epoch in range(start_epoch, epochs):
        pbar = tqdm.tqdm(total=len(train_loader), desc=f"Epoch {epoch}")
        epoch_loss = 0
        
        for data in train_loader:
            global_step += 1
            
            optimizer.zero_grad()

            # update grid every 16 steps
            if model.cuda_ray and global_step % 16 == 0:
                with torch.cuda.amp.autocast(enabled=args.fp16):
                    model.update_extra_state()
            
            # Rendering
            # Note: simplified call, using model.render via NeRFRenderer
            rays_o = data['rays_o'] # [B, N, 3]
            rays_d = data['rays_d'] # [B, N, 3]
            gt_rgb = data['images'] # [B, N, 3/4]
            
            # Use alpha channel if present
            if gt_rgb.shape[-1] == 4:
                # Alpha blending with random background
                bg_color = torch.rand_like(gt_rgb[..., :3])
                gt_rgb = gt_rgb[..., :3] * gt_rgb[..., 3:] + bg_color * (1 - gt_rgb[..., 3:])
            else:
                bg_color = 0.0 
            
            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, max_steps=args.max_steps)
                pred_rgb = outputs['image']
                
                loss = criterion(pred_rgb, gt_rgb).mean()
            
            scaler.scale(loss).backward()
            
            # Gradient clipping to prevent NaN
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            if torch.isnan(loss):
                print(f"Warning: NaN detected in loss at step {global_step}, skipping update.")
                optimizer.zero_grad()
            else:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            
            epoch_loss += loss.item()
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.6f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

        pbar.close()
        print(f"Epoch {epoch} complete, average loss: {epoch_loss/len(train_loader):.6f}")

        # Save checkpoint & Render image
        if (epoch + 1) % 20 == 0:
            ckpt_path = os.path.join(workspace, f"model_epoch_{epoch+1}.pth")
            state = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
            }
            torch.save(state, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

            # End of epoch rendering
            model.eval()
            print(f"Rendering validation image for epoch {epoch}...")
            with torch.cuda.amp.autocast(enabled=args.fp16):
                image = render_full_image(model, val_pose, val_intrinsics, val_H, val_W, bg_color=0.0, max_steps=args.max_steps)
            images.append(image)
            
            # Save PNG
            image_path = os.path.join(val_dir, f'epoch_{epoch:03d}.png')
            cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            model.train()

    # Save video
    if len(images) > 0:
        video_path = os.path.join(workspace, 'training_progress.avi')
        print(f"Generating video...")
        save_video(images, video_path)
        print(f"Video saved to {video_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./data', help="Path to data")
    parser.add_argument('--workspace', type=str, default='workspace', help="Workspace directory")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--lr', type=float, default=5e-3, help="Learning rate")
    parser.add_argument('--bound', type=float, default=2.0, help="Scene bound")
    parser.add_argument('--epochs', type=int, default=100, help="Total epochs")
    parser.add_argument('--num_rays', type=int, default=4096, help="Number of rays per batch")
    parser.add_argument('--max_steps', type=int, default=1024, help="Max steps per ray")
    parser.add_argument('--ckpt', type=str, default=None, help="Specific checkpoint to load")
    parser.add_argument('--fp16', action='store_true', help="Use Automatic Mixed Precision (AMP) for faster training")
    
    args = parser.parse_args()
    
    train(args)
