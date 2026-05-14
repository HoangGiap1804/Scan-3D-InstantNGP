import torch
import torch.optim as optim
import tqdm
import numpy as np
import os
import math
import cv2

from model import NeRFNetwork
from provider import NeRFDataset
from utils import seed_everything, render_full_image, save_video

try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False

# =============================================================================
# Metrics
# =============================================================================

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


class MetricsMeter:
    def __init__(self, device):
        self.device = device
        self.reset()
        if HAS_TORCHMETRICS:
            self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)

    def reset(self):
        self.psnr_V = 0.0
        self.ssim_V = 0.0
        self.lpips_V = 0.0
        self.N = 0

    def update(self, pred, gt, is_image=False):
        """pred, gt: torch tensor [B, N, 3] or [H, W, 3], values in [0, 1]"""
        with torch.no_grad():
            # PSNR
            mse = torch.mean((pred - gt) ** 2).item()
            psnr = -10.0 * math.log10(max(mse, 1e-10))
            self.psnr_V += psnr

            # SSIM and LPIPS — only when a full spatial image is provided
            if HAS_TORCHMETRICS and is_image:
                if pred.ndim == 3:  # [H, W, 3]
                    pred_img = pred.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
                    gt_img = gt.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
                    self.ssim_V += self.ssim(pred_img, gt_img).item()
                    self.lpips_V += self.lpips(pred_img, gt_img).item()
                elif pred.ndim == 4:  # [B, H, W, 3]
                    pred_img = pred.permute(0, 3, 1, 2).clamp(0, 1)
                    gt_img = gt.permute(0, 3, 1, 2).clamp(0, 1)
                    self.ssim_V += self.ssim(pred_img, gt_img).item()
                    self.lpips_V += self.lpips(pred_img, gt_img).item()

        self.N += 1

    def measure(self):
        return {
            'psnr': self.psnr_V / max(self.N, 1),
            'ssim': self.ssim_V / max(self.N, 1) if HAS_TORCHMETRICS else 0,
            'lpips': self.lpips_V / max(self.N, 1) if HAS_TORCHMETRICS else 0
        }

    def report(self):
        res = self.measure()
        if HAS_TORCHMETRICS and self.ssim_V > 0:
            return f"PSNR = {res['psnr']:.2f} dB, SSIM = {res['ssim']:.4f}, LPIPS = {res['lpips']:.4f}"
        return f"PSNR = {res['psnr']:.2f} dB"


# =============================================================================
# EMA
# =============================================================================

class EMA:
    def __init__(self, model, decay=0.95):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                # Byte/Bool/Int buffers (density grid, bitfield...) — copy directly
                self.shadow[k].copy_(v.detach())

    def apply_shadow(self, model):
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


# =============================================================================
# WorkspaceLogger  (mirrors torch-ngp Trainer workspace management)
# =============================================================================

import csv
import datetime

class WorkspaceLogger:
    """
    Manages workspace directory structure:
      workspace/
        checkpoints/   ← numbered .pth files
        validation/    ← rendered validation images per step
        results/       ← test-mode output images + video
        logs/
          train_log.txt  ← human-readable log
          metrics.csv    ← step,loss,psnr,lr for plotting
    """
    def __init__(self, workspace, name='ngp', max_keep_ckpt=5):
        self.workspace = workspace
        self.name = name
        self.max_keep_ckpt = max_keep_ckpt
        self._ckpt_history = []  # list of saved ckpt paths (oldest first)

        # Create subdirectories
        self.ckpt_dir  = os.path.join(workspace, 'checkpoints')
        self.val_dir   = os.path.join(workspace, 'validation')
        self.res_dir   = os.path.join(workspace, 'results')
        self.log_dir   = os.path.join(workspace, 'logs')
        for d in [self.ckpt_dir, self.val_dir, self.res_dir, self.log_dir]:
            os.makedirs(d, exist_ok=True)

        # Open log file (append mode so resume works)
        self._log_path = os.path.join(self.log_dir, 'train_log.txt')
        self._log_ptr  = open(self._log_path, 'a')

        # CSV metrics file
        self._csv_path = os.path.join(self.log_dir, 'metrics.csv')
        csv_exists = os.path.exists(self._csv_path)
        self._csv_file = open(self._csv_path, 'a', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        if not csv_exists:
            self._csv_writer.writerow(['step', 'loss', 'psnr', 'lr', 'val_psnr'])

        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.log(f"\n{'='*60}")
        self.log(f"[{ts}] WorkspaceLogger initialised → {workspace}")
        self.log(f"{'='*60}")

    def log(self, msg):
        """Print to stdout AND write to train_log.txt."""
        print(msg)
        if self._log_ptr:
            print(msg, file=self._log_ptr)
            self._log_ptr.flush()

    def log_step(self, step, loss, psnr, lr, val_psnr=None):
        """Write one row to metrics.csv."""
        self._csv_writer.writerow([step, f'{loss:.6f}', f'{psnr:.4f}', f'{lr:.2e}',
                                   f'{val_psnr:.4f}' if val_psnr is not None else ''])
        self._csv_file.flush()

    def save_checkpoint(self, state, global_step, is_best=False):
        """Save numbered checkpoint, remove old ones beyond max_keep_ckpt."""
        fname = f'{self.name}_step{global_step:07d}.pth'
        path  = os.path.join(self.ckpt_dir, fname)
        torch.save(state, path)

        # Also keep model.pth as "latest" for easy loading
        torch.save(state, os.path.join(self.workspace, 'model.pth'))

        if is_best:
            torch.save(state, os.path.join(self.workspace, 'model_best.pth'))
            self.log(f'[step {global_step}] ★ Best checkpoint saved → model_best.pth')

        self._ckpt_history.append(path)
        # Remove oldest if exceeded
        while len(self._ckpt_history) > self.max_keep_ckpt:
            old = self._ckpt_history.pop(0)
            if os.path.exists(old):
                os.remove(old)
                self.log(f'[checkpoint] Removed old checkpoint: {os.path.basename(old)}')

        self.log(f'[step {global_step}] Checkpoint saved → {fname}')
        return path

    def val_image_path(self, global_step, view_idx):
        return os.path.join(self.val_dir, f'step_{global_step:07d}_view{view_idx:02d}.png')

    def result_image_path(self, idx):
        return os.path.join(self.res_dir, f'{idx:04d}.png')

    def close(self):
        if self._log_ptr:
            self._log_ptr.close()
        if self._csv_file:
            self._csv_file.close()


# =============================================================================
# Validation helper
# =============================================================================

@torch.no_grad()
def run_validation(model, val_dataset, device, args, global_step, wl):
    """Render validation images and return metrics."""
    num_val = min(5, len(val_dataset.poses))
    val_indices = np.linspace(0, len(val_dataset.poses) - 1, num_val, dtype=int)

    # Resize to 800x800 for nicer previews
    target_H, target_W = 800, 800
    intr = val_dataset.intrinsics.copy()
    intr[0] *= target_W / val_dataset.W
    intr[1] *= target_H / val_dataset.H
    intr[2] *= target_W / val_dataset.W
    intr[3] *= target_H / val_dataset.H

    val_images = []
    metrics = MetricsMeter(device)

    with torch.amp.autocast('cuda', enabled=args.fp16):
        for vi, idx in enumerate(val_indices):
            pose = val_dataset.poses[idx:idx+1].to(device)
            img = render_full_image(
                model, pose, intr, target_H, target_W,
                bg_color=0.0, max_steps=args.max_steps,
                dt_gamma=args.dt_gamma,
                color_space=args.color_space,
            )
            val_images.append(img)
            img_path = wl.val_image_path(global_step, vi)
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # Compute metrics against GT if available
            if val_dataset.images is not None:
                gt = val_dataset.images[idx].to(device).float() / 255.0
                if gt.shape[0] != target_H or gt.shape[1] != target_W:
                    gt_np = gt.cpu().numpy()
                    gt_np = cv2.resize(gt_np, (target_W, target_H), interpolation=cv2.INTER_AREA)
                    gt = torch.from_numpy(gt_np).to(device)
                if gt.shape[-1] == 4:
                    gt = gt[..., :3] * gt[..., 3:]
                pred = torch.from_numpy(img.astype(np.float32) / 255.0).to(device)
                metrics.update(pred, gt, is_image=True)

    return val_images, metrics


# =============================================================================
# Training
# =============================================================================

def train(args):
    path = args.path
    workspace = args.workspace
    os.makedirs(workspace, exist_ok=True)

    # Workspace logger — creates subdirs + log files
    wl = WorkspaceLogger(workspace, name='ngp', max_keep_ckpt=args.max_keep_ckpt)

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 1. Load Datasets  (train + optional val split like torch-ngp)
    # -------------------------------------------------------------------------
    print(f"Loading train dataset from {path}...")
    train_dataset = NeRFDataset(
        path, type='train', device=device,
        num_rays=args.num_rays, downscale=args.downscale,
        scale=args.scale, offset=args.offset,
        color_space=args.color_space,
    )
    train_loader = train_dataset.dataloader()

    mem_images = train_dataset.images.element_size() * train_dataset.images.nelement()
    mem_poses  = train_dataset.poses.element_size()  * train_dataset.poses.nelement()
    print(f"Loaded {len(train_dataset.poses)} frames. "
          f"Dataset memory: {(mem_images + mem_poses) / 1024**2:.2f} MB")

    # Validation dataset — fall back to train poses if no val split exists
    try:
        val_dataset = NeRFDataset(
            path, type='val', device=device,
            num_rays=-1, downscale=args.downscale,
            scale=args.scale, offset=args.offset,
            color_space=args.color_space,
        )
        print(f"Loaded {len(val_dataset.poses)} validation frames.")
    except Exception:
        print("No separate val split found, using train dataset for validation.")
        val_dataset = train_dataset

    # -------------------------------------------------------------------------
    # 2. Initialize Model  (mirror torch-ngp: pass density_scale, min_near etc.)
    # -------------------------------------------------------------------------
    print(f"Initializing model with bound={args.bound}, bg_radius={args.bg_radius}...")
    model = NeRFNetwork(
        bound=args.bound,
        cuda_ray=True,
        density_scale=args.density_scale,
        min_near=args.min_near,
        density_thresh=args.density_thresh,
        bg_radius=args.bg_radius,
    ).to(device)
    print(model)

    ema = EMA(model, decay=args.ema_decay)

    # -------------------------------------------------------------------------
    # 2.5 Load Checkpoint
    # -------------------------------------------------------------------------
    ckpt_path = args.ckpt if args.ckpt else os.path.join(workspace, "model.pth")
    start_step = 0
    checkpoint = None

    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            start_step = checkpoint.get('global_step', 0)
            if 'ema' in checkpoint:
                ema.load_state_dict(checkpoint['ema'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Resuming from step {start_step}")
    else:
        print(f"No checkpoint found at {ckpt_path}, starting from scratch.")

    # -------------------------------------------------------------------------
    # 3. Optimizer & Scaler & Scheduler  (torch-ngp style: decay to 0.1 * lr)
    # -------------------------------------------------------------------------
    optimizer = optim.Adam(model.get_params(lr=args.lr), betas=(0.9, 0.99), eps=1e-15)
    scaler    = torch.cuda.amp.GradScaler(enabled=args.fp16)

    if checkpoint is not None and isinstance(checkpoint, dict):
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])

    for group in optimizer.param_groups:
        if 'initial_lr' not in group:
            group['initial_lr'] = args.lr

    total_iters = args.iters
    # Always init with last_epoch=-1 to avoid the PyTorch warning about
    # scheduler.step() being called before optimizer.step().
    # LR position is restored correctly via load_state_dict from checkpoint.
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda iter: 0.1 ** min(iter / max(total_iters, 1), 1.0),
    )
    if checkpoint is not None and isinstance(checkpoint, dict) and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    # -------------------------------------------------------------------------
    # 4. Training Loop  (iteration-based, like torch-ngp)
    # -------------------------------------------------------------------------
    model.train()
    criterion   = torch.nn.MSELoss(reduction='none')
    train_meter = MetricsMeter(device)
    progress_images = []  # for training-progress video
    best_val_psnr = -1.0

    wl.log(f'[INFO] Training {args.iters} iters | lr={args.lr} | num_rays={args.num_rays} | fp16={args.fp16}')

    if model.cuda_ray:
        model.mark_untrained_grid(train_dataset.poses, train_dataset.intrinsics)

    global_step = start_step
    loader_iter = iter(train_loader)

    pbar = tqdm.tqdm(total=total_iters, initial=global_step, desc="Training")

    while global_step < total_iters:
        # --- External control flags ---
        if os.path.exists(os.path.join(workspace, "stop.flag")):
            print("Stop flag detected. Exiting training.")
            os.remove(os.path.join(workspace, "stop.flag"))
            break

        force_save = False
        if os.path.exists(os.path.join(workspace, "save.flag")):
            force_save = True
            os.remove(os.path.join(workspace, "save.flag"))

        # --- Get next batch (cycle the loader infinitely) ---
        try:
            data = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            data = next(loader_iter)
            train_meter.reset()  # reset epoch-level metrics each pass

        global_step += 1
        optimizer.zero_grad()

        # Update occupancy grid outside autocast
        if model.cuda_ray and global_step % args.update_extra_interval == 0:
            model.update_extra_state()

        rays_o = data['rays_o']   # [B, N, 3]
        rays_d = data['rays_d']   # [B, N, 3]
        gt_rgb = data['images']   # [B, N, 3/4]

        # Random-background alpha compositing
        # If we have a background model (bg_radius > 0), we want it to learn from the data,
        # so we don't apply random augmentation.
        if args.bg_radius > 0:
            bg_color = None 
        elif gt_rgb.shape[-1] == 4:
            bg_color = torch.rand_like(gt_rgb[..., :3])
            gt_rgb = gt_rgb[..., :3] * gt_rgb[..., 3:] + bg_color * (1 - gt_rgb[..., 3:])
        else:
            bg_color = 0.0

        with torch.amp.autocast('cuda', enabled=args.fp16):
            outputs = model.render(
                rays_o, rays_d,
                staged=False,
                bg_color=bg_color,
                perturb=True,
                max_steps=args.max_steps,
                dt_gamma=args.dt_gamma,
            )
            pred_rgb = outputs['image']

            loss_per_ray = criterion(pred_rgb, gt_rgb).mean(dim=-1)  # [B, N]
            loss = loss_per_ray.mean()

            # Entropy / sparsity regularizer (anti-floater)
            if args.lambda_entropy > 0 and 'weights_sum' in outputs:
                w = outputs['weights_sum'].clamp(1e-5, 1.0 - 1e-5)
                loss_entropy = -(w * torch.log(w) + (1 - w) * torch.log(1 - w)).mean()
                loss = loss + args.lambda_entropy * loss_entropy

        # Active-error-map update
        if 'inds_coarse' in data:
            idx_tensor  = torch.tensor(data['index'], dtype=torch.long).unsqueeze(1)
            inds_coarse = data['inds_coarse'].cpu()
            err         = loss_per_ray.detach().cpu().float()
            old_err     = train_dataset.error_map[idx_tensor, inds_coarse]
            train_dataset.error_map[idx_tensor, inds_coarse] = 0.9 * old_err + 0.1 * err.clamp_min(1e-4)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if torch.isnan(loss):
            print(f"Warning: NaN loss at step {global_step}, skipping.")
            optimizer.zero_grad()
        else:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

        loss_val = loss.item()
        cur_psnr = 0.0
        with torch.no_grad():
            train_meter.update(pred_rgb.detach().clamp(0, 1), gt_rgb.clamp(0, 1), is_image=False)
            cur_psnr = train_meter.measure()['psnr']
        cur_lr = optimizer.param_groups[0]['lr']

        pbar.update(1)
        pbar.set_postfix(
            loss=f"{loss_val:.5f}",
            psnr=f"{cur_psnr:.2f}",
            lr=f"{cur_lr:.2e}",
        )

        # Log to CSV every 50 steps
        if global_step % 50 == 0:
            wl.log_step(global_step, loss_val, cur_psnr, cur_lr)

        # Build state every step — needed by both save and best-model tracking
        state = {
            'global_step': global_step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'scheduler': scheduler.state_dict(),
            'ema': ema.state_dict(),
        }

        # -----------------------------------------------------------------
        # Periodic: save checkpoint
        # -----------------------------------------------------------------
        if global_step % args.save_interval == 0 or force_save or global_step == total_iters:
            wl.save_checkpoint(state, global_step)

        # -----------------------------------------------------------------
        # Periodic: validation
        # -----------------------------------------------------------------
        if global_step % args.eval_interval == 0 or global_step == total_iters:
            model.eval()
            ema.apply_shadow(model)

            val_images, val_metrics = run_validation(
                model, val_dataset, device, args, global_step, wl
            )
            val_psnr = val_metrics.measure()['psnr']
            wl.log(f'\n[step {global_step}] Val: {val_metrics.report()}')
            wl.log_step(global_step, loss_val, cur_psnr, cur_lr, val_psnr=val_psnr)

            # Track best model
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                wl.save_checkpoint(state, global_step, is_best=True)

            if val_images:
                progress_images.append(val_images[0])

            # Restore training weights
            model.load_state_dict(state['model'])
            model.train()

    pbar.close()

    # -------------------------------------------------------------------------
    # Save training-progress video
    # -------------------------------------------------------------------------
    if progress_images:
        video_path = os.path.join(workspace, 'training_progress.mp4')
        wl.log('Generating training-progress video...')
        save_video(progress_images, video_path)
        wl.log(f'Video saved → {video_path}')

    wl.log(f'\n[DONE] Training complete. Best val PSNR = {best_val_psnr:.2f} dB')
    wl.close()


# =============================================================================
# Test / Evaluation
# =============================================================================

def test(args):
    workspace = args.workspace
    os.makedirs(workspace, exist_ok=True)

    # Use WorkspaceLogger so test results go to workspace/results/
    wl = WorkspaceLogger(workspace, name='ngp')
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wl.log(f'Using device: {device}')

    print(f"Loading test dataset from {args.path}...")
    test_dataset = NeRFDataset(
        args.path, type='test', device=device,
        downscale=args.downscale, scale=args.scale, offset=args.offset,
        color_space=args.color_space,
    )
    print(f"Loaded {len(test_dataset.poses)} test frames.")

    model = NeRFNetwork(
        bound=args.bound,
        cuda_ray=True,
        density_scale=args.density_scale,
        min_near=args.min_near,
        density_thresh=args.density_thresh,
        bg_radius=args.bg_radius,
    ).to(device)

    ckpt_path = args.ckpt if args.ckpt else os.path.join(workspace, "model.pth")
    if not os.path.exists(ckpt_path):
        # Try checkpoints/ subdir first (numbered)
        ckpt_subdir = os.path.join(workspace, 'checkpoints')
        if os.path.isdir(ckpt_subdir):
            ckpts = sorted([f for f in os.listdir(ckpt_subdir) if f.endswith('.pth')])
            if ckpts:
                ckpt_path = os.path.join(ckpt_subdir, ckpts[-1])
        if not os.path.exists(ckpt_path):
            ckpts = sorted([f for f in os.listdir(workspace) if f.endswith('.pth')])
            if ckpts:
                ckpt_path = os.path.join(workspace, ckpts[-1])

    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            weights = ckpt.get('ema', ckpt['model'])
        else:
            weights = ckpt
        model.load_state_dict(weights)
    else:
        print(f"Error: No checkpoint found at {ckpt_path}")
        return

    model.eval()
    psnr_meter = PSNRMeter()
    images_to_save = []

    print("Testing...")
    with torch.no_grad():
        for i in tqdm.trange(len(test_dataset.poses)):
            pose = test_dataset.poses[i:i+1].to(device)
            pred = render_full_image(
                model, pose, test_dataset.intrinsics,
                test_dataset.H, test_dataset.W,
                bg_color=0.0, max_steps=args.max_steps,
                dt_gamma=args.dt_gamma,
                return_float=True,
                color_space=args.color_space,
            )
            pred_t = torch.from_numpy(pred).to(device)

            if test_dataset.images is not None:
                gt = test_dataset.images[i].to(device).float() / 255.0
                if gt.shape[-1] == 4:
                    gt = gt[..., :3] * gt[..., 3:]
                psnr_meter.update(pred_t, gt)

            pred_u8 = (pred * 255).astype(np.uint8)
            images_to_save.append(pred_u8)
            cv2.imwrite(wl.result_image_path(i),
                        cv2.cvtColor(pred_u8, cv2.COLOR_RGB2BGR))

    wl.log('-' * 40)
    wl.log(f'Test complete! {psnr_meter.report()}')
    wl.log('-' * 40)

    if args.save_video:
        video_path = os.path.join(wl.res_dir, 'test_video.mp4')
        save_video(images_to_save, video_path, fps=args.fps)
        wl.log(f'Test video saved → {video_path}')

    wl.close()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # --- Positional / general ---
    parser.add_argument('--path',      type=str,   default='./data',  help="Path to dataset")
    parser.add_argument('--workspace', type=str,   default='workspace')
    parser.add_argument('--seed',      type=int,   default=0)
    parser.add_argument('--test',      action='store_true', help="Run test/eval mode instead of training")

    # --- Shortcut (mirrors torch-ngp -O) ---
    parser.add_argument('-O', action='store_true', help="Shortcut: --fp16 (AMP fast training)")

    # --- Training options ---
    parser.add_argument('--iters',     type=int,   default=30000, help="Total training iterations")
    parser.add_argument('--lr',        type=float, default=1e-2,  help="Initial learning rate")
    parser.add_argument('--ckpt',      type=str,   default=None,  help="Checkpoint path to load/resume")
    parser.add_argument('--num_rays',  type=int,   default=4096,  help="Rays sampled per training step")
    parser.add_argument('--max_steps', type=int,   default=1024,  help="Max marching steps per ray (cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16,
                        help="Steps between density-grid updates")
    parser.add_argument('--fp16',      action='store_true', help="Use AMP mixed precision")
    parser.add_argument('--ema_decay', type=float, default=0.95,  help="EMA decay (0 = off)")
    parser.add_argument('--save_interval', type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument('--eval_interval',  type=int, default=500,  help="Run validation every N steps")
    parser.add_argument('--lambda_entropy', type=float, default=1e-3,
                        help="Weight for entropy/sparsity loss (anti-floater)")
    parser.add_argument('--max_keep_ckpt', type=int, default=5,
                        help="Max numbered checkpoints to keep in checkpoints/ dir")

    # --- Dataset options (mirrors torch-ngp) ---
    parser.add_argument('--downscale', type=int,   default=1,     help="Downscale images")
    parser.add_argument('--scale',     type=float, default=0.33,  help="Scale camera positions into [-bound, bound]^3")
    parser.add_argument('--offset',    type=float, nargs='*', default=[0, 0, 0], help="Camera position offset")
    parser.add_argument('--color_space', type=str, default='srgb', choices=['srgb', 'linear'], help="Color space of images")

    # --- Scene / model options (mirrors torch-ngp) ---
    parser.add_argument('--bound',         type=float, default=2,    help="Scene bound (box half-size)")
    parser.add_argument('--bg_radius',     type=float, default=-1,   help="Background sphere radius (>0 enables)")
    parser.add_argument('--dt_gamma',      type=float, default=1/128,help="Adaptive step size gamma (0 = disabled)")
    parser.add_argument('--min_near',      type=float, default=0.2,  help="Minimum near distance")
    parser.add_argument('--density_thresh',type=float, default=10,   help="Density threshold for occupancy grid")
    parser.add_argument('--density_scale', type=float, default=1,    help="Scale applied to sigma values")

    # --- Test-mode options ---
    parser.add_argument('--save_video', action='store_true', help="Save test output as video")
    parser.add_argument('--fps',        type=int, default=30)

    args = parser.parse_args()

    # -O shortcut
    if args.O:
        args.fp16 = True

    print(args)

    if args.test:
        test(args)
    else:
        train(args)
