"""
nerf_server.py  ─  Headless NeRF Render Server
================================================
Chạy NeRF model ở chế độ không có GUI.
Lắng nghe kết nối TCP từ Blender add-on, nhận camera pose,
render ảnh và trả về PNG bytes.

Cách dùng:
    python nerf_server.py /path/to/dataset_dir \\
        --workspace workspaces/my_model \\
        --test -O --bound 1.0 --scale 0.7 \\
        --dt_gamma 0 --color_space linear \\
        --host 127.0.0.1 --port 6789

Giao thức (mỗi message = 4-byte big-endian uint32 length + payload):
    Request  (Blender → Server): JSON bytes
    Response (Server → Blender): PNG  bytes
"""

import torch
import argparse
import socketserver
import threading
import struct
import json
import io
import time
import numpy as np

from nerf.provider import NeRFDataset
from nerf.utils import *

# ──────────────────────────────────────────────────────────────────────────────
# TCP Protocol helpers
# ──────────────────────────────────────────────────────────────────────────────

def _recv_exactly(sock, n: int) -> bytes | None:
    """Đọc đúng n bytes từ socket. Trả về None nếu kết nối đóng."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_framed(sock) -> bytes | None:
    """Đọc một message có tiền tố 4-byte độ dài."""
    header = _recv_exactly(sock, 4)
    if header is None:
        return None
    length = struct.unpack(">I", header)[0]
    return _recv_exactly(sock, length)


def send_framed(sock, data: bytes) -> None:
    """Gửi một message có tiền tố 4-byte độ dài."""
    sock.sendall(struct.pack(">I", len(data)) + data)


# ──────────────────────────────────────────────────────────────────────────────
# PNG encode helpers
# ──────────────────────────────────────────────────────────────────────────────

def encode_png(image: np.ndarray) -> bytes:
    """
    Chuyen numpy array [H, W, 3] hoac [H, W, 4] float32 -> PNG bytes.
    Uu tien Pillow, fallback sang OpenCV.
    """
    image_u8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    channels = image.shape[2] if image.ndim == 3 else 1
    try:
        from PIL import Image
        mode = "RGBA" if channels == 4 else "RGB"
        buf = io.BytesIO()
        Image.fromarray(image_u8, mode).save(buf, format="PNG",
                                             optimize=False, compress_level=1)
        return buf.getvalue()
    except ImportError:
        import cv2
        if channels == 4:
            bgra = cv2.cvtColor(image_u8, cv2.COLOR_RGBA2BGRA)
            ok, enc = cv2.imencode(".png", bgra, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        else:
            bgr = cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)
            ok, enc = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return enc.tobytes()


# ──────────────────────────────────────────────────────────────────────────────
# Render helpers
# ──────────────────────────────────────────────────────────────────────────────

def render_frame(trainer, pose, intrinsics, W, H, bg_color_tensor, transparent=False):
    """
    Render mot frame tu NeRF model.

    transparent=False: tra ve [H, W, 3] float32  (RGB, co nen)
    transparent=True:  tra ve [H, W, 4] float32  (RGBA, alpha = weights_sum)

    Khi transparent=True, render voi nen den (bg=0) va trich xuat weights_sum
    tu model.render() lam kenh alpha. Dung premultiplied alpha:
        RGB = foreground * alpha  (vi render voi nen den)
        A   = weights_sum
    """
    from nerf.utils import get_rays

    pose_t = torch.from_numpy(pose).unsqueeze(0).to(trainer.device)  # [1,4,4]
    rays   = get_rays(pose_t, intrinsics, H, W, -1)
    data   = {'rays_o': rays['rays_o'], 'rays_d': rays['rays_d'], 'H': H, 'W': W}

    trainer.model.eval()
    if trainer.ema is not None:
        trainer.ema.store()
        trainer.ema.copy_to()

    # Neu transparent, render voi nen den de lay RGB phan canh nguyen chat
    if transparent:
        render_bg = torch.zeros(3, dtype=torch.float32).to(trainer.device)
    else:
        render_bg = bg_color_tensor.to(trainer.device) if bg_color_tensor is not None else None

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=trainer.fp16):
            outputs = trainer.model.render(
                data['rays_o'], data['rays_d'],
                staged=True,
                bg_color=render_bg,
                perturb=False,
                **vars(trainer.opt)
            )

    if trainer.ema is not None:
        trainer.ema.restore()

    pred_rgb = outputs['image'].reshape(H, W, 3)

    if transparent:
        if 'weights_sum' in outputs:
            alpha_tensor = outputs['weights_sum'].reshape(H, W, 1)
        else:
            alpha_tensor = outputs['image'].reshape(H, W, 3).max(dim=2, keepdim=True)[0]
            
        pred_rgb_tensor = outputs['image'].reshape(H, W, 3)
        
        # Un-premultiply in torch
        safe_alpha = torch.clamp(alpha_tensor, 1e-6, 1.0)
        pred_rgb_straight = pred_rgb_tensor / safe_alpha
        
        if trainer.opt.color_space == 'linear':
            pred_rgb_straight = linear_to_srgb(pred_rgb_straight)
            
        # Convert to numpy
        pred_rgb_straight_np = pred_rgb_straight.detach().cpu().numpy()
        alpha_np = alpha_tensor.detach().cpu().numpy()
            
        # Boost alpha slightly to make the object solid
        alpha_boosted = np.clip(alpha_np * 1.5, 0.0, 1.0).astype(np.float32)
        
        # Re-premultiply with boosted alpha for Blender's ALPHA_PREMULT
        pred_rgb_premult = pred_rgb_straight_np * alpha_boosted
        
        rgba = np.concatenate([pred_rgb_premult, alpha_boosted], axis=2)
        return rgba
    else:
        pred_rgb = outputs['image'].reshape(H, W, 3)
        if trainer.opt.color_space == 'linear':
            pred_rgb = linear_to_srgb(pred_rgb)
        return pred_rgb.detach().cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Request Handler
# ──────────────────────────────────────────────────────────────────────────────

class NeRFHandler(socketserver.StreamRequestHandler):
    """
    Xử lý một kết nối TCP persistent.
    Client (Blender) giữ kết nối và gửi nhiều request liên tiếp.
    """

    def handle(self):
        addr = self.client_address
        print(f"\n[Server] ✔  Client kết nối: {addr[0]}:{addr[1]}")
        frame_count = 0
        total_ms = 0.0

        while True:
            try:
                raw = recv_framed(self.request)
                if raw is None:
                    break  # Client đóng kết nối

                req         = json.loads(raw.decode("utf-8"))
                pose        = np.array(req["pose"],       dtype=np.float32)  # [4,4]
                intrinsics  = np.array(req["intrinsics"], dtype=np.float32)  # [4]
                W           = int(req["W"])
                H           = int(req["H"])
                bg          = req.get("bg_color", [1.0, 1.0, 1.0])
                transparent = bool(req.get("transparent", False))
                bg_tensor   = torch.tensor(bg, dtype=torch.float32)

                t0 = time.perf_counter()
                with self.server.gpu_lock:
                    image = render_frame(
                        self.server.trainer,
                        pose, intrinsics, W, H,
                        bg_color_tensor=bg_tensor,
                        transparent=transparent,
                    )
                dt_ms = (time.perf_counter() - t0) * 1000.0

                png_bytes = encode_png(image)
                send_framed(self.request, png_bytes)

                # ── Log ────────────────────────────────────────────────────
                frame_count += 1
                total_ms += dt_ms
                avg_fps = 1000.0 / (total_ms / frame_count) if total_ms > 0 else 0
                print(
                    f"\r[Server] frame={frame_count:5d}  "
                    f"{W}x{H}  "
                    f"{dt_ms:6.1f}ms  "
                    f"avg {avg_fps:5.1f} FPS   ",
                    end="",
                    flush=True,
                )

            except (ConnectionResetError, BrokenPipeError, OSError):
                break
            except Exception as exc:
                import traceback
                print(f"\n[Server] Lỗi khi xử lý request: {exc}")
                traceback.print_exc()
                break

        print(f"\n[Server] ✘  Client ngắt kết nối: {addr[0]}:{addr[1]}")


# ──────────────────────────────────────────────────────────────────────────────
# TCP Server
# ──────────────────────────────────────────────────────────────────────────────

class NeRFTCPServer(socketserver.ThreadingTCPServer):
    """
    ThreadingTCPServer: mỗi client connection chạy trong thread riêng.
    gpu_lock đảm bảo chỉ có một luồng dùng GPU tại một thời điểm.
    """
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, handler_class, trainer):
        self.trainer = trainer
        self.gpu_lock = threading.Lock()
        super().__init__(server_address, handler_class)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Headless NeRF Render Server – gửi ảnh render cho Blender qua TCP"
    )

    # ── Dataset / model (giống main_nerf.py) ──────────────────────────────────
    parser.add_argument("path", type=str, help="Đường dẫn đến thư mục chứa transforms.json")
    parser.add_argument("-O", action="store_true", help="fp16 + cuda_ray + preload")
    parser.add_argument("--test", action="store_true", help="Chỉ load checkpoint, không train")
    parser.add_argument("--workspace", type=str, default="workspace")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default="latest")

    # Training (bỏ qua nếu --test)
    parser.add_argument("--iters", type=int, default=30000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num_rays", type=int, default=4096)
    parser.add_argument("--cuda_ray", action="store_true")
    parser.add_argument("--max_steps", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=512)
    parser.add_argument("--upsample_steps", type=int, default=0)
    parser.add_argument("--update_extra_interval", type=int, default=16)
    parser.add_argument("--max_ray_batch", type=int, default=4096)
    parser.add_argument("--patch_size", type=int, default=1)

    # Dataset
    parser.add_argument("--color_space", type=str, default="srgb")
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--downscale", type=int, default=1)
    parser.add_argument("--bound", type=float, default=2)
    parser.add_argument("--scale", type=float, default=0.33)
    parser.add_argument("--offset", nargs="*", type=float, default=[0, 0, 0])
    parser.add_argument("--dt_gamma", type=float, default=1 / 128)
    parser.add_argument("--min_near", type=float, default=0.2)
    parser.add_argument("--density_thresh", type=float, default=10)
    parser.add_argument("--bg_radius", type=float, default=-1)
    parser.add_argument("--error_map", action="store_true")
    parser.add_argument("--clip_text", type=str, default="")
    parser.add_argument("--rand_pose", type=int, default=-1)

    # GUI (không dùng, nhưng Trainer cần opt.gui, opt.W, opt.H)
    parser.add_argument("--gui", action="store_true", default=False)
    parser.add_argument("--W", type=int, default=800)
    parser.add_argument("--H", type=int, default=800)
    parser.add_argument("--radius", type=float, default=5)
    parser.add_argument("--fovy", type=float, default=50)
    parser.add_argument("--max_spp", type=int, default=64)

    # ── Server options ─────────────────────────────────────────────────────────
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Địa chỉ lắng nghe (mặc định: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=6789,
                        help="Cổng lắng nghe (mặc định: 6789)")

    opt = parser.parse_args()

    # ── Flags shortcuts ────────────────────────────────────────────────────────
    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True
    else:
        opt.fp16 = False

    if opt.patch_size > 1:
        opt.error_map = False
        assert opt.num_rays % (opt.patch_size ** 2) == 0

    # ── Bắt buộc chạy ở test mode ─────────────────────────────────────────────
    if not opt.test:
        print("[Server] Cảnh báo: --test không được chỉ định. Server tự chuyển sang test mode.")
        opt.test = True

    seed_everything(opt.seed)

    # ── Tạo model ──────────────────────────────────────────────────────────────
    from nerf.network_tcnn import NeRFNetwork

    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.MSELoss(reduction="none")

    metrics = [PSNRMeter(), LPIPSMeter(device=device)]
    trainer = Trainer(
        "ngp",
        opt,
        model,
        device=device,
        workspace=opt.workspace,
        criterion=criterion,
        fp16=opt.fp16,
        metrics=metrics,
        use_checkpoint=opt.ckpt,
    )

    print(f"\n[Server] Model đã load xong!")
    print(f"[Server] Khởi động TCP server tại {opt.host}:{opt.port} ...")
    print(f"[Server] Nhấn Ctrl+C để dừng.\n")

    server = NeRFTCPServer((opt.host, opt.port), NeRFHandler, trainer)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Đang dừng...")
        server.shutdown()
        print("[Server] Đã dừng.")
