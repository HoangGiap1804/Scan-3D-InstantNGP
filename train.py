import torch
import torch.optim as optim
import tqdm
import numpy as np
import os
import math

from model import NeRFNetwork
from provider import NeRFDataset
from utils import seed_everything, render_full_image, save_video
import cv2

class PSNRMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.V = 0.0
        self.N = 0

    def update(self, pred, gt):
        """pred, gt: torch tensor [*, 3], values in [0, 1]"""
        with torch.no_grad():
            mse = torch.mean((pred - gt) ** 2).item()
            psnr = -10.0 * math.log10(max(mse, 1e-10))
        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / max(self.N, 1)

    def report(self):
        return f"PSNR = {self.measure():.2f} dB"

class EMA:
    def __init__(self, model, decay=0.95):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                # Chỉ EMA trên float tensors (parameters & float buffers)
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                # Byte/Bool/Int buffers (density grid, bitfield...) — copy trực tiếp
                self.shadow[k].copy_(v.detach())

    def apply_shadow(self, model):
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}

def train(args):
    # Configuration
    path = args.path
    workspace = args.workspace
    os.makedirs(workspace, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 1. Load Dataset
    # -------------------------------------------------------------------------
    print(f"Loading dataset from {path}...")
    train_dataset = NeRFDataset(path, type='train', device=device, num_rays=args.num_rays, downscale=args.downscale)
    train_loader = train_dataset.dataloader()
    
    # Print dataset memory usage
    mem_images = train_dataset.images.element_size() * train_dataset.images.nelement()
    mem_poses = train_dataset.poses.element_size() * train_dataset.poses.nelement()
    total_mem = (mem_images + mem_poses) / (1024 * 1024) # MB
    print(f"Loaded {len(train_dataset.poses)} frames. Dataset memory usage: {total_mem:.2f} MB")

    # -------------------------------------------------------------------------
    # 2. Initialize Model
    # -------------------------------------------------------------------------
    print(f"Initializing model with bound {args.bound}...")
    model = NeRFNetwork(bound=args.bound, cuda_ray=True).to(device)
    print(model)

    # EMA wrapper (decay=0.95 giống torch-ngp)
    ema = EMA(model, decay=args.ema_decay)

    # -------------------------------------------------------------------------
    # 2.5 Load Checkpoint
    # -------------------------------------------------------------------------
    ckpt_path = args.ckpt if args.ckpt else os.path.join(workspace, "model.pth")
    start_epoch = 0
    checkpoint = None

    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)

        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint.get('epoch', 0)
            if 'ema' in checkpoint:
                ema.load_state_dict(checkpoint['ema'])
        else:
            # Định dạng cũ: chỉ có state_dict
            model.load_state_dict(checkpoint)
        print(f"Resuming from epoch {start_epoch}")
    else:
        print(f"No checkpoint found at {ckpt_path}, starting from scratch.")

    print("Starting training...")
    epochs = args.epochs
    global_step = start_epoch * len(train_loader)

    # 3. Optimizer & Scaler
    # -------------------------------------------------------------------------
    optimizer = optim.Adam(model.get_params(lr=args.lr), betas=(0.9, 0.99), eps=1e-15)

    # FIX: Dùng torch.amp thay vì torch.cuda.amp (tránh deprecated API)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # Load optimizer & scaler state nếu có (PHẢI TRƯỚC KHI TẠO SCHEDULER)
    if checkpoint is not None and isinstance(checkpoint, dict):
        if 'optimizer' in checkpoint:
            print("Loading optimizer state...")
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])

    # FIX: Đảm bảo có initial_lr trong mỗi param_group để tránh KeyError khi resume
    for group in optimizer.param_groups:
        if 'initial_lr' not in group:
            group['initial_lr'] = args.lr

    # FIX: Thêm min() để clamp LR, tránh LR tiếp tục giảm sau max_steps
    max_steps = epochs * len(train_loader)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda iter: 0.1 ** min(iter / max(max_steps, 1), 1.0),
        # FIX: last_epoch=global_step thay thế vòng lặp fast-forward
        last_epoch=global_step - 1 if global_step > 0 else -1,
    )

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    model.train()
    criterion = torch.nn.MSELoss(reduction='none')
    psnr_meter = PSNRMeter()

    # Validation setup
    val_dir = os.path.join(workspace, 'validation')
    os.makedirs(val_dir, exist_ok=True)
    images = []

    # FIX: Dùng nhiều val poses để đánh giá toàn diện hơn (tối đa 5 poses)
    num_val_poses = min(5, len(train_dataset.poses))
    val_indices = np.linspace(0, len(train_dataset.poses) - 1, num_val_poses, dtype=int)
    val_poses = train_dataset.poses[val_indices].to(device)
    val_intrinsics = train_dataset.intrinsics
    val_H, val_W = train_dataset.H, train_dataset.W

    # Mark untrained grid region
    if model.cuda_ray:
        model.mark_untrained_grid(train_dataset.poses, train_dataset.intrinsics)

    for epoch in range(start_epoch, epochs):
        # External Control: Check for stop flag
        stop_flag = os.path.join(workspace, "stop.flag")
        if os.path.exists(stop_flag):
            print(f"External Stop signal detected at {stop_flag}. Exiting training...")
            os.remove(stop_flag)
            break
            
        # External Control: Check for manual save flag
        save_flag = os.path.join(workspace, "save.flag")
        force_save = False
        if os.path.exists(save_flag):
            print(f"External Save signal detected at {save_flag}.")
            force_save = True
            os.remove(save_flag)

        pbar = tqdm.tqdm(total=len(train_loader), desc=f"Epoch {epoch}")
        epoch_loss = 0.0
        psnr_meter.reset()

        for data in train_loader:
            global_step += 1

            optimizer.zero_grad()

            # FIX: update_extra_state() NGOÀI autocast — không cần mixed precision
            if model.cuda_ray and global_step % args.update_extra_interval == 0:
                model.update_extra_state()

            rays_o = data['rays_o']   # [B, N, 3]
            rays_d = data['rays_d']   # [B, N, 3]
            gt_rgb = data['images']   # [B, N, 3/4]

            # Alpha blending với random background nếu có alpha channel
            if gt_rgb.shape[-1] == 4:
                bg_color = torch.rand_like(gt_rgb[..., :3])
                gt_rgb = gt_rgb[..., :3] * gt_rgb[..., 3:] + bg_color * (1 - gt_rgb[..., 3:])
            else:
                bg_color = 0.0

            # FIX: Dùng torch.amp.autocast thay vì torch.cuda.amp.autocast
            with torch.amp.autocast('cuda', enabled=args.fp16):
                outputs = model.render(
                    rays_o, rays_d,
                    staged=False,
                    bg_color=bg_color,
                    perturb=True,
                    max_steps=args.max_steps,
                )
                pred_rgb = outputs['image']
                loss = criterion(pred_rgb, gt_rgb).mean()

            scaler.scale(loss).backward()

            # Gradient clipping — ngăn NaN/exploding gradient
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if torch.isnan(loss):
                print(f"Warning: NaN in loss at step {global_step}, skipping update.")
                optimizer.zero_grad()
            else:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # EMA update sau mỗi optimizer step
                ema.update(model)

            # Track metrics
            loss_val = loss.item()
            epoch_loss += loss_val
            with torch.no_grad():
                psnr_meter.update(pred_rgb.detach().clamp(0, 1), gt_rgb.clamp(0, 1))

            pbar.update(1)
            pbar.set_postfix(
                loss=f"{loss_val:.6f}",
                psnr=f"{psnr_meter.measure():.2f}dB",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        pbar.close()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} complete | loss: {avg_loss:.6f} | {psnr_meter.report()}")

        # ---------------------------------------------------------------------
        # Save checkpoint & Render validation images
        # ---------------------------------------------------------------------
        if (epoch + 1) % args.save_interval == 0 or force_save:
            ckpt_save_path = os.path.join(workspace, "model.pth")
            state = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'ema': ema.state_dict(),
            }
            torch.save(state, ckpt_save_path)
            print(f"Saved checkpoint to {ckpt_save_path}")

            # Render validation image (dùng EMA weights để ảnh mượt hơn)
            model.eval()
            ema.apply_shadow(model)
            print(f"Rendering {num_val_poses} validation image(s) for epoch {epoch}...")

            val_images_epoch = []
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=args.fp16):
                    for vi, vpose in enumerate(val_poses):
                        img = render_full_image(
                            model, vpose.unsqueeze(0),
                            val_intrinsics, val_H, val_W,
                            bg_color=0.0, max_steps=args.max_steps,
                        )
                        val_images_epoch.append(img)
                        image_path = os.path.join(val_dir, f'epoch_{epoch:03d}_view{vi:02d}.png')
                        cv2.imwrite(image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # Dùng view đầu tiên để tạo video progress
            images.append(val_images_epoch[0])

            # Khôi phục model weights gốc để tiếp tục train
            model.load_state_dict(state['model'])
            model.train()

    # -------------------------------------------------------------------------
    # Save training progress video
    # -------------------------------------------------------------------------
    if len(images) > 0:
        video_path = os.path.join(workspace, 'training_progress.mp4')
        print("Generating training progress video...")
        save_video(images, video_path)
        print(f"Video saved to {video_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./data', help="Path to data")
    parser.add_argument('--workspace', type=str, default='workspace', help="Workspace directory")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--lr', type=float, default=1e-2, help="Initial learning rate")
    parser.add_argument('--bound', type=float, default=0.5, help="Scene bound")
    parser.add_argument('--epochs', type=int, default=100, help="Total epochs")
    parser.add_argument('--num_rays', type=int, default=2048, help="Number of rays per batch")
    parser.add_argument('--max_steps', type=int, default=1024, help="Max steps per ray (cuda_ray)")
    parser.add_argument('--ckpt', type=str, default=None, help="Specific checkpoint path to load")
    parser.add_argument('--fp16', action='store_true', help="Use AMP (fp16) for faster training")
    parser.add_argument('--ema_decay', type=float, default=0.95, help="EMA decay rate (0 = disable)")
    parser.add_argument('--save_interval', type=int, default=20, help="Save checkpoint every N epochs")
    parser.add_argument('--downscale', type=int, default=1, help="Downscale images before loading")
    parser.add_argument('--update_extra_interval', type=int, default=16,
                        help="Update density grid every N steps (cuda_ray)")

    args = parser.parse_args()
    train(args)
