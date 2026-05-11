import torch
import numpy as np
import os
import cv2
import dearpygui.dearpygui as dpg
import threading
import queue
import time
import asyncio
import websockets
import base64
import json
import socket
import qrcode

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
        self.t_thresh = 0.001 # Ngưỡng dừng tia (Tăng để nhanh hơn)
        self.dt_gamma = 0     # Bước nhảy thích ứng
        self.model = None
        self.image = np.zeros((H, W, 4), dtype=np.float32)
        self.image[..., 3] = 1.0 # Set opaque alpha once
        self.intrinsics = None
        
        # Thread safety flags
        self.new_image_ready = False
        self.current_fps = 0.0
        
        # Background Encoder Queue
        self.encode_queue = queue.Queue(maxsize=1)
        self.encoder_thread = threading.Thread(target=self._encoder_worker, daemon=True)
        self.encoder_thread.start()
        
        # Interaction state
        self.is_moving = False
        self.last_move_time = 0
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
            dpg.add_slider_float(label="T Thresh", min_value=0.0, max_value=0.1, format="%.4f", default_value=self.t_thresh, callback=self.set_t_thresh)
            dpg.add_slider_float(label="dt Gamma", min_value=0.0, max_value=0.1, format="%.4f", default_value=self.dt_gamma, callback=self.set_dt_gamma)
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
    def set_t_thresh(self, sender, data): self.t_thresh = data; self.need_update = True
    def set_dt_gamma(self, sender, data): self.dt_gamma = data; self.need_update = True
    def force_update(self, sender, data): self.need_update = True

    def on_mouse_wheel(self, sender, app_data):
        if dpg.is_item_hovered("_primary_window"):
            self.radius -= app_data * 0.5
            self.radius = np.clip(self.radius, 0.1, 10.0)
            dpg.set_value("_radius_slider", self.radius)
            self.need_update = True
            self.is_moving = True
            self.last_move_time = time.time()

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
        ws_port = getattr(self, 'port_ws', 8000)
        
        server_url = f"ws://{local_ip}:{ws_port}"
        
        print(f"\n" + "="*50)
        print(f"THÔNG TIN KẾT NỐI SERVER")
        print(f"Viewer (WS): {server_url}")
        print(f"-"*50)
        try:
            qr = qrcode.QRCode(version=1, box_size=1, border=1)
            qr.add_data(local_ip) # Chỉ chứa địa chỉ IP
            qr.make(fit=True)
            print(f"Quét mã QR để lấy IP Server ({local_ip}):")
            qr.print_ascii(invert=True)
        except Exception as e:
            print(f"Could not generate QR code: {e}")
        print("="*50 + "\n")

        # --- WebSocket Handler ---
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
            # Chạy WebSocket Server
            ws_server = websockets.serve(handler, "0.0.0.0", ws_port)
            
            print(f"[SERVER] WebSocket chạy tại cổng {ws_port}")

            await asyncio.gather(
                ws_server,
                self._ws_broadcast_task()
            )

        try:
            self.loop.run_until_complete(main())
        except Exception as e:
            print(f"[SERVER] Error: {e}")
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
                        dpg.set_value("_elevation_slider", self.elevation)
                        self.is_moving = True
                        self.last_move_time = time.time()
                        self.need_update = True
                mouse_last_pos = mouse_pos
                mouse_was_down = True
            else:
                # If not dragging, check if we should stop moving (for zoom or after release)
                if self.is_moving and time.time() - self.last_move_time > 0.1:
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
                
                # Dynamic rendering quality & Resolution
                if self.is_moving:
                    render_H, render_W = 100, 100
                    current_steps = 16
                    current_upsample = 0
                else:
                    render_H, render_W = self.H, self.W
                    current_steps = self.num_steps
                    current_upsample = self.upsample_steps
                
                # Scale intrinsics based on current render resolution
                s_H = render_H / self.H
                s_W = render_W / self.W
                curr_intrinsics = intrinsics.copy()
                curr_intrinsics[0] *= s_W # fl_x
                curr_intrinsics[1] *= s_H # fl_y
                curr_intrinsics[2] *= s_W # cx
                curr_intrinsics[3] *= s_H # cy
                
                # Render using float32 directly
                with torch.no_grad():
                    image_float = render_full_image(
                        self.model, pose, curr_intrinsics, render_H, render_W, 
                        bg_color=self.bg_color,
                        return_float=True,
                        num_steps=current_steps, 
                        upsample_steps=current_upsample,
                        T_thresh=self.t_thresh,
                        dt_gamma=self.dt_gamma
                    )
                
                # Upscale if rendering at lower resolution
                if render_H != self.H or render_W != self.W:
                    image_float = cv2.resize(image_float, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
                    
                # Update texture without creating new numpy array
                self.image[..., :3] = image_float
                
                # FPS update
                now = time.time()
                self.current_fps = 1.0 / (now - last_time)
                last_time = now
                
                # Signal main thread to update UI
                self.new_image_ready = True

                # Offload JPEG encoding to background thread to not block rendering
                if (self.clients or self.last_packet is None) and not self.encode_queue.full():
                    # Copy image data to avoid race conditions
                    img_to_encode = self.image[..., :3].copy()
                    self.encode_queue.put_nowait((img_to_encode, self.current_fps))
            else:
                time.sleep(0.001) # Reduced sleep for faster response

    def _encoder_worker(self):
        """Background thread to handle JPEG encoding and WebSocket preparation"""
        while self.running:
            try:
                # Wait for new image to encode
                img_float, fps = self.encode_queue.get(timeout=1.0)
                
                # Encode to JPEG
                img_uint8 = (img_float * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                
                # Add metadata
                packet = json.dumps({
                    "type": "image",
                    "signature": "InstantNGP",
                    "image": jpg_as_text,
                    "fps": fps,
                    "res": f"{self.W}x{self.H}"
                })
                self.last_packet = packet
                
                # Push to broadcast queue
                if self.clients and not self.ws_queue.full():
                    self.ws_queue.put_nowait(packet)
                    
            except queue.Empty:
                continue
            except Exception as e:
                # print(f"[Encoder] Error: {e}")
                pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--res', type=int, default=400, help="Render resolution")
    parser.add_argument('--display', type=int, default=None, help="Display resolution (upscale)")
    parser.add_argument('--angle', type=float, default=None, help="Camera angle x (FOV) override")
    parser.add_argument('--port_ws', type=int, default=8000)
    args = parser.parse_args()
    
    # Pass ports to GUI
    GUI.port_ws = args.port_ws

    gui = GUI(args.workspace, args.ckpt, H=args.res, W=args.res, camera_angle_x=args.angle, display_res=args.display)
    gui.render_loop()
