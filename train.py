import torch
import torch.optim as optim
import tqdm
import numpy as np
import os

from model import NeRFNetwork
from provider import NeRFDataset
from utils import seed_everything

def train():
    # Configuration
    path = "./data" # Path to your data folder containing transforms_train.json
    workspace = "workspace"
    os.makedirs(workspace, exist_ok=True)
    
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Dataset
    print("Loading dataset...")
    train_dataset = NeRFDataset(path, type='train', device=device)
    train_loader = train_dataset.dataloader()
    print(f"Loaded {len(train_dataset.poses)} frames.")

    # 2. Initialize Model
    print("Initializing model...")
    model = NeRFNetwork(bound=2, cuda_ray=False).to(device)
    print(model)

    # 3. Optimizer & Scheduler
    optimizer = optim.Adam(model.get_params(lr=1e-2), betas=(0.9, 0.99), eps=1e-15)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / 1000, 1))

    # 4. Training Loop
    model.train()
    criterion = torch.nn.MSELoss(reduction='none')

    print("Starting training...")
    epochs = 10
    global_step = 0

    for epoch in range(epochs):
        pbar = tqdm.tqdm(total=len(train_loader), desc=f"Epoch {epoch}")
        epoch_loss = 0
        
        for data in train_loader:
            global_step += 1
            
            optimizer.zero_grad()
            
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
                bg_color = 1.0 # White background
            
            outputs = model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True)
            pred_rgb = outputs['image']
            
            loss = criterion(pred_rgb, gt_rgb).mean()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.6f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

        pbar.close()
        print(f"Epoch {epoch} complete, average loss: {epoch_loss/len(train_loader):.6f}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(workspace, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    train()
