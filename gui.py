import torch
import numpy as np
import os
import cv2
import dearpygui.dearpygui as dpg
import threading
import time

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
        self.intrinsics = None
        
        # Load model and potentially intrinsics
        self.load_model(ckpt_path)
        self.try_load_intrinsics()
        
        # DearPyGui Setup
        dpg.create_context()
        self.setup_dpg()
        
    def load_model(self, ckpt_path=None):
        print(f"Initializing model...")
        self.model = NeRFNetwork(bound=0.5, cuda_ray=True).to(self.device).eval()
        
        if ckpt_path is None:
            # Find latest checkpoint in workspace
            ckpts = [f for f in os.listdir(self.workspace) if f.endswith('.pth')]
            if ckpts:
                ckpts.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if 'epoch' in x else 0)
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
            dpg.add_slider_float(label="Azimuth", min_value=0, max_value=360, default_value=self.azimuth, callback=self.set_azimuth)
            dpg.add_slider_float(label="Elevation", min_value=-90, max_value=90, default_value=self.elevation, callback=self.set_elevation)
            dpg.add_slider_float(label="Radius", min_value=0.1, max_value=10.0, default_value=self.radius, callback=self.set_radius)
            
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

    # Callbacks
    def set_azimuth(self, sender, data): self.azimuth = data; self.need_update = True
    def set_elevation(self, sender, data): self.elevation = data; self.need_update = True
    def set_radius(self, sender, data): self.radius = data; self.need_update = True
    def set_bg_color(self, sender, data): self.bg_color = data; self.need_update = True
    def set_steps(self, sender, data): self.num_steps = data; self.need_update = True
    def set_upsample_steps(self, sender, data): self.upsample_steps = data; self.need_update = True
    def force_update(self, sender, data): self.need_update = True

    def render_loop(self):
        last_time = time.time()
        while dpg.is_dearpygui_running():
            if self.need_update:
                self.need_update = False
                
                # Calculate pose
                pose = get_orbit_pose(self.azimuth, self.elevation, self.radius, self.center)
                pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)
                
                # Mock intrinsics (matching train.py setup or typical values)
                # In a real app, these should come from the dataset/config
                fl = self.W  # Default 90 deg FOV if W=H
                intrinsics = np.array([fl, fl, self.W / 2, self.H / 2])
                
                # Render
                with torch.no_grad():
                    # Use quality parameters
                    image_uint8 = render_full_image(
                        self.model, pose, self.intrinsics, self.H, self.W, 
                        bg_color=self.bg_color, 
                        num_steps=self.num_steps, 
                        upsample_steps=self.upsample_steps
                    )
                    
                # Fix flipped image (vertical and horizontal flip)
                image_uint8 = np.flipud(np.fliplr(image_uint8))
                
                # Update texture (convert uint8 to float32 [0, 1] for DPG)
                image_float = image_uint8.astype(np.float32) / 255.0
                
                # Convert RGB to RGBA for DPG (RGBA expects 4 floats per pixel)
                rgba = np.zeros((self.H, self.W, 4), dtype=np.float32)
                rgba[..., :3] = image_float
                rgba[..., 3] = 1.0 # Opaque alpha
                
                dpg.set_value("_texture", rgba.flatten())
                
                # FPS update
                now = time.time()
                fps = 1.0 / (now - last_time)
                last_time = now
                dpg.set_value(self.fps_text, f"FPS: {fps:.2f}")
            
            dpg.render_dearpygui_frame()

        dpg.destroy_context()

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
