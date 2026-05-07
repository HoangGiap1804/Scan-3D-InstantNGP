import torch
import numpy as np
import os
import cv2
import dearpygui.dearpygui as dpg
import threading
import time
import asyncio
import websockets
import base64
import json
import socket
import qrcode
import queue

from model import NeRFNetwork
from utils import seed_everything, render_full_image, get_orbit_pose, linear_to_srgb

class GUI:
    def __init__(self, workspace, ckpt_path=None, H=800, W=800, camera_angle_x=None, display_res=None):
        self.workspace = workspace
        self.H = H
        self.W = W
        self.display_W = display_res if display_res else W
        self.display_H = display_res if display_res else H
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Camera parameters
        self.azimuth = 0
        self.elevation = 0
        self.radius = 4.0
        self.center = np.array([0, 0, 0], dtype=np.float32)
        self.camera_angle_x_override = camera_angle_x
        
        # Rendering state
        self.need_update = True
        self.bg_color = 0.0
        self.num_steps = 128
        self.upsample_steps = 128
        self.model = None
        self.image = np.zeros((H, W, 4), dtype=np.float32)
        self.image[..., 3] = 1.0 # Set opaque alpha once
        self.intrinsics = None
        
        # Thread safety flags
        self.new_image_ready = False
        self.current_fps = 0.0
        
        # Interaction state
        self.is_moving = False
        self.running = True
        
        # Load model and potentially intrinsics
        self.load_model(ckpt_path)
        self.try_load_intrinsics()
        
        # WebSocket state
        self.clients = set()
        self.ws_queue = queue.Queue(maxsize=2) # Keep queue small to avoid lag
        self.last_packet = None
        self.loop = None
        
        # Start WebSocket server thread
        self.ws_thread = threading.Thread(target=self._start_ws_server, daemon=True)
        self.ws_thread.start()

        # DearPyGui Setup
        dpg.create_context()
        self.setup_dpg()
        
    def load_model(self, ckpt_path=None):
        print(f"Initializing model...")
        self.model = NeRFNetwork(bound=0.5, cuda_ray=True).to(self.device).eval()
        
        if ckpt_path is None:
            # Ưu tiên tìm model.pth
            ckpt_path = os.path.join(self.workspace, "model.pth")
            
            # Nếu không có model.pth mới tìm các file .pth khác
            if not os.path.exists(ckpt_path):
                ckpts = [f for f in os.listdir(self.workspace) if f.endswith('.pth')]
                if ckpts:
                    def get_epoch(name):
                        try:
                            return int(name.split('_')[-1].split('.')[0])
                        except ValueError:
                            return 999999
                    ckpts.sort(key=get_epoch)
                    ckpt_path = os.path.join(self.workspace, ckpts[-1])
        
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"Loading checkpoint from {ckpt_path}...")
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            print("No checkpoint found, using uninitialized model.")

    def try_load_intrinsics(self):
        # Use manual override if provided
        if self.camera_angle_x_override is not None:
            fl_x = self.W / (2 * np.tan(self.camera_angle_x_override / 2))
            fl_y = fl_x
            self.intrinsics = np.array([fl_x, fl_y, self.W / 2, self.H / 2])
            print(f"Using camera_angle_x override: {self.camera_angle_x_override}")
            return

        # Try to find transforms.json to get real intrinsics
        json_paths = [
            os.path.join(self.workspace, 'transforms_train.json'),
            os.path.join(self.workspace, 'transforms.json'),
            os.path.join(os.path.dirname(self.workspace), 'data', 'transforms_train.json'),
        ]
        for path in json_paths:
            if os.path.exists(path):
                try:
                    import json
                    with open(path, 'r') as f:
                        transform = json.load(f)
                    if 'camera_angle_x' in transform:
                        fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2))
                        fl_y = fl_x
                    elif 'fl_x' in transform:
                        scale = self.W / transform['w'] if 'w' in transform else 1.0
                        fl_x = transform['fl_x'] * scale
                        fl_y = transform['fl_y'] * scale
                    else:
                        continue
                    
                    cx = (transform['cx'] * (self.W / transform['w'])) if 'cx' in transform else (self.W / 2)
                    cy = (transform['cy'] * (self.H / transform['h'])) if 'cy' in transform else (self.H / 2)
                    
                    self.intrinsics = np.array([fl_x, fl_y, cx, cy])
                    print(f"Loaded intrinsics from {path}")
                    return
                except Exception as e:
                    print(f"Failed to load intrinsics from {path}: {e}")
        
        # Fallback
        fl = self.W
        self.intrinsics = np.array([fl, fl, self.W / 2, self.H / 2])
        print("Using default intrinsics.")

    def setup_dpg(self):
        # Create dynamic texture (at render resolution)
        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(width=self.W, height=self.H, default_value=self.image.flatten(), tag="_texture")

        # Main Window (Image Area - occupy the full space)
        with dpg.window(tag="_primary_window", width=self.display_W, height=self.display_H, no_scrollbar=True, no_move=True):
            # Upscale image to display resolution
            dpg.add_image("_texture", width=self.display_W, height=self.display_H)
            
        # Control Panel (Separate Floating Window)
        with dpg.window(label="Instant-NGP Controls", tag="_control_window", width=280, height=500, pos=(5, 5)):
            dpg.add_text("Instant-NGP Viewer", color=(100, 200, 255))
            dpg.add_separator()
            
            dpg.add_text("Camera Controls")
            dpg.add_slider_float(label="Azimuth", min_value=0, max_value=360, default_value=self.azimuth, callback=self.set_azimuth, tag="_azimuth_slider")
            dpg.add_slider_float(label="Elevation", min_value=-90, max_value=90, default_value=self.elevation, callback=self.set_elevation, tag="_elevation_slider")
            dpg.add_slider_float(label="Radius", min_value=0.1, max_value=10.0, default_value=self.radius, callback=self.set_radius, tag="_radius_slider")
            
            dpg.add_separator()
            dpg.add_text("Rendering Options")
            dpg.add_slider_float(label="Background Color", min_value=0.0, max_value=1.0, default_value=self.bg_color, callback=self.set_bg_color)
            dpg.add_slider_int(label="Steps", min_value=16, max_value=1024, default_value=self.num_steps, callback=self.set_steps)
            dpg.add_slider_int(label="Upsample Steps", min_value=0, max_value=512, default_value=self.upsample_steps, callback=self.set_upsample_steps)
            dpg.add_button(label="Force Update", callback=self.force_update)
            
            dpg.add_separator()
            dpg.add_text("Information")
            dpg.add_text(f"Device: {self.device}")
            dpg.add_text(f"Render Res: {self.W}x{self.H}")
            dpg.add_text(f"Display Res: {self.display_W}x{self.display_H}")
            self.fps_text = dpg.add_text("FPS: N/A")

        dpg.create_viewport(title='Instant-NGP Interactive Viewer', width=self.display_W, height=self.display_H)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("_primary_window", True)
        
        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=self.on_mouse_wheel)

    # Callbacks
    def set_azimuth(self, sender, data): self.azimuth = data; self.need_update = True
    def set_elevation(self, sender, data): self.elevation = data; self.need_update = True
    def set_radius(self, sender, data): self.radius = data; self.need_update = True
    def set_bg_color(self, sender, data): self.bg_color = data; self.need_update = True
    def set_steps(self, sender, data): self.num_steps = data; self.need_update = True
    def set_upsample_steps(self, sender, data): self.upsample_steps = data; self.need_update = True
    def force_update(self, sender, data): self.need_update = True

    def on_mouse_wheel(self, sender, app_data):
        if dpg.is_item_hovered("_primary_window"):
            self.radius -= app_data * 0.5
            self.radius = np.clip(self.radius, 0.1, 10.0)
            dpg.set_value("_radius_slider", self.radius)
            self.need_update = True
            self.is_moving = False

    def _start_ws_server(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Get local IP for QR code
        def get_local_ip():
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(('8.8.8.8', 80))
                ip = s.getsockname()[0]
            except Exception:
                ip = '127.0.0.1'
            finally:
                s.close()
            return ip

        local_ip = get_local_ip()
        port = 8000
        server_url = f"ws://{local_ip}:{port}"
        
        print(f"\n" + "="*50)
        print(f"[WS] Server Address: {server_url}")
        try:
            qr = qrcode.QRCode(version=1, box_size=1, border=1)
            qr.add_data(server_url)
            qr.make(fit=True)
            print("[WS] Scan QR code to connect:")
            qr.print_ascii(invert=True)
        except Exception as e:
            print(f"[WS] Could not generate QR code: {e}")
        print("="*50 + "\n")

        async def handler(websocket):
            self.clients.add(websocket)
            print(f"[WS] Client connected. Total: {len(self.clients)}")
            
            # Gửi ảnh mới nhất ngay khi kết nối
            if self.last_packet:
                await websocket.send(self.last_packet)

            try:
                # Lắng nghe tin nhắn từ Client
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        if data.get("type") == "camera":
                            # Cập nhật thông số camera
                            if "azimuth" in data: self.azimuth = float(data["azimuth"])
                            if "elevation" in data: self.elevation = float(data["elevation"])
                            if "radius" in data: self.radius = float(data["radius"])
                            
                            self.need_update = True
                            
                            # Đồng bộ ngược lại giao diện DearPyGui (nếu đang mở)
                            dpg.set_value("_azimuth_slider", self.azimuth)
                            dpg.set_value("_elevation_slider", self.elevation)
                            dpg.set_value("_radius_slider", self.radius)

                            # Gửi ngay ảnh hiện tại để phản hồi nhanh
                            if self.last_packet:
                                await websocket.send(self.last_packet)
                    except Exception as e:
                        print(f"[WS] Error processing message: {e}")
            except websockets.ConnectionClosed:
                pass
            finally:
                if websocket in self.clients:
                    self.clients.remove(websocket)
                print(f"[WS] Client disconnected. Total: {len(self.clients)}")

        async def main():
            # Khởi tạo server trong async context để tránh lỗi "no running event loop"
            async with websockets.serve(handler, "0.0.0.0", 8000):
                print("[WS] Server started on ws://0.0.0.0:8000")
                # Chạy task broadcast ngay trong loop này
                await self._ws_broadcast_task()

        try:
            self.loop.run_until_complete(main())
        except Exception as e:
            print(f"[WS] Server error: {e}")
        finally:
            self.loop.close()

    async def _ws_broadcast_task(self):
        while self.running:
            try:
                # Lấy dữ liệu từ queue (không block event loop)
                if not self.ws_queue.empty():
                    data = await self.loop.run_in_executor(None, self.ws_queue.get)
                    if self.clients:
                        # Gửi đến tất cả client đang kết nối
                        # Sử dụng wait thay vì gather để tránh crash nếu client ngắt kết nối đột ngột
                        active_clients = list(self.clients)
                        if active_clients:
                            await asyncio.wait([asyncio.create_task(client.send(data)) for client in active_clients], timeout=0.1)
                else:
                    await asyncio.sleep(0.01) # Tránh chiếm dụng CPU khi queue trống
            except Exception as e:
                # print(f"[WS] Broadcast error: {e}")
                await asyncio.sleep(0.1)

    def render_loop(self):
        self.render_thread = threading.Thread(target=self._render_worker, daemon=True)
        self.render_thread.start()
        
        mouse_last_pos = (0, 0)
        mouse_was_down = False
        
        while dpg.is_dearpygui_running():
            # Mouse drag interaction
            is_down = dpg.is_mouse_button_down(dpg.mvMouseButton_Left)
            is_hovered = dpg.is_item_hovered("_primary_window")
            
            if is_down and (is_hovered or mouse_was_down):
                mouse_pos = dpg.get_mouse_pos(local=False)
                if mouse_was_down:
                    dx = mouse_pos[0] - mouse_last_pos[0]
                    dy = mouse_pos[1] - mouse_last_pos[1]
                    if dx != 0 or dy != 0:
                        self.azimuth -= dx * 0.5
                        self.elevation += dy * 0.5
                        self.elevation = np.clip(self.elevation, -89.0, 89.0)
                        dpg.set_value("_azimuth_slider", self.azimuth)
                        dpg.set_value("_elevation_slider", self.elevation)
                        self.is_moving = True
                        self.need_update = True
                mouse_last_pos = mouse_pos
                mouse_was_down = True
            else:
                if mouse_was_down:
                    # Mouse released, trigger high quality render
                    self.is_moving = False
                    self.need_update = True
                mouse_was_down = False
            
            # Thread-safe UI update
            if self.new_image_ready:
                dpg.set_value("_texture", self.image.flatten())
                if self.current_fps > 0:
                    dpg.set_value(self.fps_text, f"FPS: {self.current_fps:.2f}")
                self.new_image_ready = False
            
            dpg.render_dearpygui_frame()

        self.running = False
        self.render_thread.join()
        dpg.destroy_context()

    def _render_worker(self):
        last_time = time.time()
        while self.running:
            if self.need_update:
                self.need_update = False
                
                # Calculate pose
                pose = get_orbit_pose(self.azimuth, self.elevation, self.radius, self.center)
                pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)
                
                fl = self.W  # Default FOV if unspecified
                intrinsics = self.intrinsics if self.intrinsics is not None else np.array([fl, fl, self.W / 2, self.H / 2])
                
                # Dynamic rendering quality
                current_steps = 16 if self.is_moving else self.num_steps
                current_upsample = 0 if self.is_moving else self.upsample_steps
                
                # Render using float32 directly
                with torch.no_grad():
                    image_float = render_full_image(
                        self.model, pose, intrinsics, self.H, self.W, 
                        bg_color=self.bg_color,
                        return_float=True,
                        num_steps=current_steps, 
                        upsample_steps=current_upsample
                    )
                    
                # Update texture without creating new numpy array
                self.image[..., :3] = image_float
                
                # FPS update
                now = time.time()
                self.current_fps = 1.0 / (now - last_time)
                last_time = now
                
                # Signal main thread to update UI
                self.new_image_ready = True

                # Signal WebSocket to broadcast (Encode if clients connected OR if no packet exists yet)
                if (self.clients or self.last_packet is None) and not self.ws_queue.full():
                    try:
                        # Encode to JPEG
                        img_uint8 = (self.image[..., :3] * 255).astype(np.uint8)
                        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                        _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                        
                        # Add metadata
                        packet = json.dumps({
                            "type": "image",
                            "signature": "InstantNGP",
                            "image": jpg_as_text,
                            "fps": self.current_fps,
                            "res": f"{self.W}x{self.H}"
                        })
                        self.last_packet = packet
                        if self.clients:
                            self.ws_queue.put_nowait(packet)
                    except queue.Full:
                        pass
            else:
                time.sleep(0.005)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--res', type=int, default=400, help="Render resolution")
    parser.add_argument('--display', type=int, default=None, help="Display resolution (upscale)")
    parser.add_argument('--angle', type=float, default=None, help="Camera angle x (FOV) override")
    args = parser.parse_args()

    gui = GUI(args.workspace, args.ckpt, H=args.res, W=args.res, camera_angle_x=args.angle, display_res=args.display)
    gui.render_loop()
