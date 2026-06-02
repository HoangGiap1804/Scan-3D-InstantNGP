"""
blender_nerf_addon.py  ─  NeRF Live Render Add-on for Blender
==============================================================
Display rendered images from NeRF Server directly in Blender Viewport.

Installation:
    Edit > Preferences > Add-ons > Install...
    Select this file -> Enable "NeRF Live Render"

Usage:
    1. Run nerf_server.py first
    2. In the 3D Viewport, open the N-panel (press N) -> "NeRF" tab
    3. Enter host/port -> press "Start NeRF View"
    4. Rotate/zoom Blender viewport -> NeRF image updates accordingly

Blender Requirement: 3.0+
"""

bl_info = {
    "name": "NeRF Live Render",
    "author": "NeRF Project",
    "version": (1, 1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > N-Panel > NeRF",
    "description": "Connect with NeRF Server to render and display directly in the Viewport",
    "category": "Render",
}

import bpy
import gpu
import math
import socket
import struct
import json
import io
import threading
import time
import numpy as np

from gpu_extras.batch import batch_for_shader


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

TIMER_INTERVAL  = 0.033   # ~30 FPS target for modal timer
REQUEST_TIMEOUT = 5.0     # socket timeout (seconds)
MAX_RENDER_DIM  = 1024    # max render dimension


# ──────────────────────────────────────────────────────────────────────────────
# TCP Protocol helpers
# ──────────────────────────────────────────────────────────────────────────────

def _recv_exactly(sock, n):
    buf = b""
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except (socket.timeout, OSError):
            return None
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_framed(sock):
    header = _recv_exactly(sock, 4)
    if header is None:
        return None
    length = struct.unpack(">I", header)[0]
    return _recv_exactly(sock, length)


def send_framed(sock, data):
    sock.sendall(struct.pack(">I", len(data)) + data)


# ──────────────────────────────────────────────────────────────────────────────
# Camera helpers  --  EXACTLY like nerf_matrix_to_ngp() in provider.py
# ──────────────────────────────────────────────────────────────────────────────

def blender_to_nerf_matrix(pose_blender, scale, offset):
    """
    Convert camera-to-world from Blender to NeRF (instant-ngp) coordinate system.

    This is EXACTLY the nerf_matrix_to_ngp() transformation in provider.py:

      Blender / transforms.json:   X=right, Y=up,      Z=backward  (OpenGL)
      NeRF (ngp):                  X=right, Y=forward, Z=up

    Row mapping:
      NeRF row 0 <- Blender row Y (index 1),  inverted sign on columns 1 and 2
      NeRF row 1 <- Blender row Z (index 2),  inverted sign on columns 1 and 2
      NeRF row 2 <- Blender row X (index 0),  inverted sign on columns 1 and 2

    Translation is multiplied by scale and offset is added.
    """
    p = pose_blender
    return np.array([
        [ p[1, 0], -p[1, 1], -p[1, 2],  p[1, 3] * scale + offset[0]],
        [ p[2, 0], -p[2, 1], -p[2, 2],  p[2, 3] * scale + offset[1]],
        [ p[0, 0], -p[0, 1], -p[0, 2],  p[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)


def get_blender_camera_pose(region_3d, scale=1.0, offset=None):
    """
    Read camera pose from Blender viewport and convert to NeRF coordinate system.

    Blender mathutils.Matrix is row-major.
    view_matrix = world-to-camera  ->  invert to get camera-to-world.
    """
    if offset is None:
        offset = [0.0, 0.0, 0.0]

    view = np.array(region_3d.view_matrix, dtype=np.float32).reshape(4, 4)
    pose_blender = np.linalg.inv(view)   # camera-to-world, Blender convention
    return blender_to_nerf_matrix(pose_blender, scale=scale, offset=offset)


def get_blender_intrinsics(region_3d, W, H):
    """
    Calculate [fx, fy, cx, cy] from Blender viewport's projection matrix.
    This guarantees exact FOV matching, preventing sliding during panning.
    """
    P = np.array(region_3d.window_matrix, dtype=np.float32)
    
    fx = (W / 2.0) * P[0, 0]
    fy = (H / 2.0) * P[1, 1]

    return np.array([fx, fy, W / 2.0, H / 2.0], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# GPU Texture helper
# ──────────────────────────────────────────────────────────────────────────────

def numpy_to_gpu_texture(image):
    """
    Convert [H, W, 3] or [H, W, 4] float32 -> GPUTexture.
    Flip Y because Blender uses bottom-left coordinate system.
    NeRF returns top-left origin.
    """
    H, W   = image.shape[:2]
    is_rgba = image.ndim == 3 and image.shape[2] == 4
    img     = image[::-1, :, :].copy()   # flip Y

    if is_rgba:
        # Already has alpha in 4th channel
        rgba = img.astype(np.float32)
    else:
        # Add alpha = 1.0
        rgba      = np.ones((H, W, 4), dtype=np.float32)
        rgba[:, :, :3] = img

    flat = rgba.flatten()
    buf  = gpu.types.Buffer("FLOAT", len(flat), flat)
    return gpu.types.GPUTexture((W, H), format="RGBA32F", data=buf)


# ──────────────────────────────────────────────────────────────────────────────
# PNG decode
# ──────────────────────────────────────────────────────────────────────────────

def decode_png_to_numpy(png_bytes):
    """PNG bytes -> [H, W, 3] or [H, W, 4] float32 [0,1]"""
    try:
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(png_bytes))
        # Keep RGBA if present (transparent render)
        if img.mode == 'RGBA':
            arr = np.array(img.convert('RGBA'), dtype=np.float32) / 255.0  # [H,W,4]
        else:
            arr = np.array(img.convert('RGB'),  dtype=np.float32) / 255.0  # [H,W,3]
        return arr
    except ImportError:
        import cv2
        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError("cv2.imdecode failed")
        if img.shape[2] == 4:  # BGRA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:                   # BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0


# ──────────────────────────────────────────────────────────────────────────────
# Shared State between Network Thread and Blender UI Thread
# ──────────────────────────────────────────────────────────────────────────────

class SharedState:
    """Thread-safe data sharing between network thread and UI thread."""

    def __init__(self):
        self._lock   = threading.Lock()
        self.running = False

        # Camera data (set by UI thread)
        self.latest_pose       = None
        self.latest_intrinsics = None
        self.render_W          = 512
        self.render_H          = 512
        self.bg_color          = [1.0, 1.0, 1.0]
        self.transparent       = False   # whether to call RGBA render or RGB

        # Latest rendered image (set by network thread)
        self.latest_image  = None
        self.image_updated = False
        self.image_w       = 512   # actual pixel width of last render
        self.image_h       = 512   # actual pixel height of last render

        # Status
        self.fps    = 0.0
        self.status = "Disconnected"
        self.error  = ""

    def set_camera(self, pose, intrinsics, W, H, bg, transparent=False):
        with self._lock:
            self.latest_pose       = pose
            self.latest_intrinsics = intrinsics
            self.render_W          = W
            self.render_H          = H
            self.bg_color          = bg
            self.transparent       = transparent

    def get_camera(self):
        with self._lock:
            return (self.latest_pose, self.latest_intrinsics,
                    self.render_W, self.render_H, self.bg_color, self.transparent)

    def set_image(self, image):
        with self._lock:
            self.latest_image  = image
            self.image_updated = True
            self.image_h, self.image_w = image.shape[:2]

    def get_image_if_new(self):
        with self._lock:
            if self.image_updated:
                self.image_updated = False
                return self.latest_image, self.image_w, self.image_h
            return None, None, None

    def get_image_size(self):
        with self._lock:
            return self.image_w, self.image_h

    def set_status(self, status, error="", fps=0.0):
        with self._lock:
            self.status = status
            self.error  = error
            self.fps    = fps


# ──────────────────────────────────────────────────────────────────────────────
# Network Thread
# ──────────────────────────────────────────────────────────────────────────────

def network_thread_fn(host, port, state):
    """
    Runs in background thread.
    Connects to NeRF server, sends requests continuously and receives rendered images.
    """
    sock = None
    try:
        state.set_status("Connecting...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(REQUEST_TIMEOUT)
        sock.connect((host, port))
        sock.settimeout(REQUEST_TIMEOUT)
        state.set_status(f"Connected to {host}:{port}")
        print(f"[NeRF Addon] Connected to {host}:{port}")

        t_prev = time.perf_counter()

        while state.running:
            pose, intrinsics, W, H, bg, transparent = state.get_camera()

            if pose is None:
                time.sleep(0.01)
                continue

            req     = {"pose": pose.tolist(), "intrinsics": intrinsics.tolist(),
                       "W": W, "H": H, "bg_color": bg, "transparent": transparent}
            payload = json.dumps(req).encode("utf-8")
            send_framed(sock, payload)

            png_bytes = recv_framed(sock)
            if png_bytes is None:
                state.set_status("Error: Connection lost", error="Server closed connection")
                break

            image = decode_png_to_numpy(png_bytes)
            state.set_image(image)

            t_now = time.perf_counter()
            fps   = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now
            state.set_status(f"Rendering | {W}x{H}", fps=fps)

    except ConnectionRefusedError:
        state.set_status("Error: Connection refused",
                         error=f"Could not connect to {host}:{port}. Is the server running?")
    except socket.timeout:
        state.set_status("Error: Timeout", error="Server did not respond")
    except Exception as exc:
        state.set_status(f"Error: {type(exc).__name__}", error=str(exc))
        import traceback; traceback.print_exc()
    finally:
        if sock:
            try: sock.close()
            except Exception: pass
        state.running = False
        print("[NeRF Addon] Network thread stopped.")


# ──────────────────────────────────────────────────────────────────────────────
# Draw Callback
# ──────────────────────────────────────────────────────────────────────────────

def get_image_shader():
    for name in ("IMAGE", "2D_IMAGE"):
        try:
            return gpu.shader.from_builtin(name)
        except Exception:
            continue
    raise RuntimeError("IMAGE shader not found")


def build_quad_batch(shader, x, y, w, h):
    verts   = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
    uvs     = [(0, 0), (1,   0), (1,   1   ), (0, 1   )]
    indices = [(0, 1, 2), (0, 2, 3)]
    return batch_for_shader(shader, "TRIS",
                            {"pos": verts, "texCoord": uvs},
                            indices=indices)


# ──────────────────────────────────────────────────────────────────────────────
# Global draw state
# ──────────────────────────────────────────────────────────────────────────────

_active_state     = None
_draw_handle      = None
_gpu_texture      = None
_shader           = None
_batch            = None
_last_draw_key    = None
_is_transparent_mode = False   # track current blend mode


def _compute_draw_rect(rw, rh, iw, ih, mode):
    """
    Calculate (x, y, w, h) to draw image onto the viewport.

    mode:
      'FILL'   - stretch image to always fill viewport (legacy behavior)
      'FIT'    - preserve aspect ratio, fit inside viewport (default)
      'CORNER' - display in bottom-right corner at original size (iw x ih)
    """
    if mode == 'FILL':
        return 0, 0, rw, rh

    elif mode == 'FIT':
        # Keep aspect ratio, not larger than viewport
        scale = min(rw / iw, rh / ih)
        dw = int(iw * scale)
        dh = int(ih * scale)
        x  = (rw - dw) // 2
        y  = (rh - dh) // 2
        return x, y, dw, dh

    else:  # 'CORNER'
        # Display at bottom-right corner, original size (iw x ih)
        # If too large, downscale to fit viewport
        dw = min(iw, rw)
        dh = min(ih, rh)
        x  = rw - dw
        y  = 0
        return x, y, dw, dh


def draw_nerf_viewport(context):
    """Draw callback: draw NeRF image onto viewport (POST_PIXEL)."""
    global _active_state, _gpu_texture, _shader, _batch, _last_draw_key

    if _active_state is None or _gpu_texture is None:
        return

    props   = context.scene.nerf_props
    opacity = props.opacity
    mode    = props.display_mode
    region  = context.region
    rw, rh  = region.width, region.height
    iw, ih  = _active_state.get_image_size()

    if _shader is None:
        try:
            _shader = get_image_shader()
        except Exception as e:
            print(f"[NeRF Addon] Shader error: {e}")
            return

    draw_key = (rw, rh, iw, ih, mode)
    if draw_key != _last_draw_key:
        x, y, dw, dh = _compute_draw_rect(rw, rh, iw, ih, mode)
        _batch        = build_quad_batch(_shader, x, y, dw, dh)
        _last_draw_key = draw_key

    if _batch is None:
        x, y, dw, dh = _compute_draw_rect(rw, rh, iw, ih, mode)
        _batch = build_quad_batch(_shader, x, y, dw, dh)

    gpu.state.blend_set("ALPHA_PREMULT" if _is_transparent_mode else "ALPHA")
    _shader.bind()
    try:
        _shader.uniform_float("color", (1.0, 1.0, 1.0, opacity))
    except Exception:
        pass
    _shader.uniform_sampler("image", _gpu_texture)
    _batch.draw(_shader)
    gpu.state.blend_set("NONE")


# ──────────────────────────────────────────────────────────────────────────────
# Operators
# ──────────────────────────────────────────────────────────────────────────────

class NERF_OT_StartLiveRender(bpy.types.Operator):
    """Start NeRF Server connection and display render in viewport"""
    bl_idname  = "nerf.start_live_render"
    bl_label   = "Start NeRF View"
    bl_options = {"REGISTER"}

    _timer = None

    def modal(self, context, event):
        global _active_state, _gpu_texture, _shader, _batch, _last_region_size

        if _active_state is None or not _active_state.running:
            self.cancel(context)
            return {"CANCELLED"}

        if event.type == "TIMER":
            props     = context.scene.nerf_props
            region_3d = None
            space_3d  = None
            region_win = None

            for area in context.screen.areas:
                if area.type == "VIEW_3D":
                    for region in area.regions:
                        if region.type == "WINDOW":
                            space = area.spaces.active
                            if hasattr(space, "region_3d"):
                                region_3d  = space.region_3d
                                region_win = region
                                space_3d   = space
                                break
                    if region_3d:
                        break

            if region_3d is None:
                return {"PASS_THROUGH"}

            render_W = min(props.render_width,  MAX_RENDER_DIM)
            render_H = min(props.render_height, MAX_RENDER_DIM)

            # If using viewport size, override render W/H
            if props.use_viewport_size:
                ds = max(0.1, min(1.0, props.viewport_downscale))
                render_W = max(64, int(region_win.width  * ds))
                render_H = max(64, int(region_win.height * ds))

            # Get camera pose using exact provider.py transformation
            scale  = props.nerf_scale
            offset = [props.nerf_offset_x, props.nerf_offset_y, props.nerf_offset_z]
            pose   = get_blender_camera_pose(region_3d, scale=scale, offset=offset)

            intrinsics = get_blender_intrinsics(region_3d, render_W, render_H)

            bg_r, bg_g, bg_b = props.bg_color
            _active_state.set_camera(pose, intrinsics, render_W, render_H,
                                     [bg_r, bg_g, bg_b],
                                     transparent=props.transparent)

            new_image, iw, ih = _active_state.get_image_if_new()
            if new_image is not None:
                _gpu_texture      = numpy_to_gpu_texture(new_image)
                _is_transparent_mode = (new_image.ndim == 3 and new_image.shape[2] == 4)
                _shader           = None
                _batch            = None
                _last_draw_key    = None
                for area in context.screen.areas:
                    if area.type == "VIEW_3D":
                        area.tag_redraw()

        return {"PASS_THROUGH"}

    def invoke(self, context, event):
        global _active_state, _draw_handle, _gpu_texture, _shader, _batch

        props = context.scene.nerf_props
        host  = props.host
        port  = props.port

        _active_state = SharedState()
        _active_state.running = True
        _gpu_texture = None
        _shader      = None
        _batch       = None

        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            draw_nerf_viewport, (context,), "WINDOW", "POST_PIXEL"
        )

        t = threading.Thread(
            target=network_thread_fn,
            args=(host, port, _active_state),
            daemon=True,
        )
        t.start()

        self._timer = context.window_manager.event_timer_add(
            TIMER_INTERVAL, window=context.window
        )
        context.window_manager.modal_handler_add(self)

        props.is_running = True
        print(f"[NeRF Addon] Starting. Connecting to {host}:{port}...")
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        global _active_state, _draw_handle, _gpu_texture

        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

        if _draw_handle:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, "WINDOW")
            _draw_handle = None

        if _active_state:
            _active_state.running = False
            _active_state = None

        _gpu_texture = None

        context.scene.nerf_props.is_running = False
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


class NERF_OT_StopLiveRender(bpy.types.Operator):
    """Stop NeRF connection and remove overlay from viewport"""
    bl_idname  = "nerf.stop_live_render"
    bl_label   = "Stop"
    bl_options = {"REGISTER"}

    def execute(self, context):
        global _active_state, _draw_handle, _gpu_texture

        if _active_state:
            _active_state.running = False
        if _draw_handle:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, "WINDOW")
            _draw_handle = None
        _gpu_texture = None

        context.scene.nerf_props.is_running = False
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()

        self.report({"INFO"}, "NeRF Live Render stopped")
        return {"FINISHED"}


# ──────────────────────────────────────────────────────────────────────────────
# Properties
# ──────────────────────────────────────────────────────────────────────────────

class NeRFProperties(bpy.types.PropertyGroup):
    host: bpy.props.StringProperty(
        name="Host", default="127.0.0.1",
        description="IP address of NeRF server",
    )
    port: bpy.props.IntProperty(
        name="Port", default=6789, min=1024, max=65535,
        description="TCP port of NeRF server",
    )
    render_width: bpy.props.IntProperty(
        name="W", default=512, min=64, max=MAX_RENDER_DIM,
        description="Width (only used when 'Use Viewport Size' is disabled)",
    )
    render_height: bpy.props.IntProperty(
        name="H", default=512, min=64, max=MAX_RENDER_DIM,
        description="Height (only used when 'Use Viewport Size' is disabled)",
    )

    # === RENDER SIZE AUTO ===
    use_viewport_size: bpy.props.BoolProperty(
        name="Use Viewport Size",
        description="Automatically use Viewport size as render resolution",
        default=True,
    )
    viewport_downscale: bpy.props.FloatProperty(
        name="Downscale",
        description="Downscale factor: 1.0 = full resolution, 0.5 = half size (increases FPS)",
        default=0.5, min=0.1, max=1.0, step=5, precision=2, subtype="FACTOR",
    )

    # === SCALE & OFFSET ===
    nerf_scale: bpy.props.FloatProperty(
        name="--scale",
        description=(
            "The --scale value when running nerf_server.py (e.g., 0.7).\n"
            "Must match exactly so Blender units align with NeRF space.\n"
            "Default: 0.33 (fox dataset), this project uses 0.7"
        ),
        default=0.7, min=0.001, max=10.0, step=1, precision=4,
    )
    nerf_offset_x: bpy.props.FloatProperty(
        name="Offset X",
        description="The --offset[0] value when training NeRF",
        default=0.0, step=1, precision=3,
    )
    nerf_offset_y: bpy.props.FloatProperty(
        name="Offset Y",
        description="The --offset[1] value when training NeRF",
        default=0.0, step=1, precision=3,
    )
    nerf_offset_z: bpy.props.FloatProperty(
        name="Offset Z",
        description="The --offset[2] value when training NeRF",
        default=0.0, step=1, precision=3,
    )

    # === BACKGROUND / TRANSPARENCY ===
    transparent: bpy.props.BoolProperty(
        name="Transparent Background",
        description=(
            "Render with transparent background (RGBA). "
            "NeRF renders with black background and uses weights_sum as actual alpha. "
            "Disable to use a custom background color."
        ),
        default=True,
    )
    bg_color: bpy.props.FloatVectorProperty(
        name="Background Color", default=(1.0, 1.0, 1.0),
        min=0.0, max=1.0, subtype="COLOR",
        description="Background color (only used when Transparent Background = False)",
    )

    # === DISPLAY ===
    display_mode: bpy.props.EnumProperty(
        name="Display",
        description="How to display NeRF images in the viewport",
        items=[
            ('FILL',   "Fill Viewport (FILL)", "Fill entire viewport — recommended with Viewport Size"),
            ('FIT',    "Fit Window (FIT)",    "Keep aspect ratio, fit inside viewport, centered"),
            ('CORNER', "Corner of Screen",     "Bottom-right corner, original size"),
        ],
        default='FILL',
    )
    opacity: bpy.props.FloatProperty(
        name="Opacity", default=1.0, min=0.0, max=1.0, subtype="FACTOR",
        description="Opacity of the NeRF image overlay",
    )
    is_running: bpy.props.BoolProperty(name="Is Running", default=False)


# ──────────────────────────────────────────────────────────────────────────────
# Panel
# ──────────────────────────────────────────────────────────────────────────────

class VIEW3D_PT_NeRFLiveRender(bpy.types.Panel):
    bl_label      = "NeRF Live Render"
    bl_idname     = "VIEW3D_PT_nerf_live_render"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category   = "NeRF"

    def draw(self, context):
        layout = self.layout
        props  = context.scene.nerf_props

        # ── Server connection ─────────────────────────────────────────────────
        box = layout.box()
        box.label(text="Server Connection", icon="NETWORK_DRIVE")
        col = box.column(align=True)
        col.prop(props, "host")
        col.prop(props, "port")

        # ── Render size ───────────────────────────────────────────────────────
        box = layout.box()
        box.label(text="Render Size", icon="IMAGE_DATA")
        col = box.column(align=True)
        col.prop(props, "use_viewport_size")
        if props.use_viewport_size:
            col.prop(props, "viewport_downscale", slider=True)
            col.label(text="Auto: viewport x downscale", icon="INFO")
        else:
            row = col.row(align=True)
            row.prop(props, "render_width")
            row.prop(props, "render_height")

        # ── Scale / Offset ────────────────────────────────────────────────────
        box = layout.box()
        box.label(text="Align Units with NeRF", icon="ORIENTATION_GLOBAL")
        col = box.column(align=True)
        col.prop(props, "nerf_scale")
        row = col.row(align=True)
        row.label(text="Offset:")
        row.prop(props, "nerf_offset_x", text="X")
        row.prop(props, "nerf_offset_y", text="Y")
        row.prop(props, "nerf_offset_z", text="Z")

        # ── Background & Display ──────────────────────────────────────────────
        box = layout.box()
        box.label(text="Background & Display", icon="RENDERLAYERS")
        col = box.column(align=True)
        col.prop(props, "transparent")
        if not props.transparent:
            col.prop(props, "bg_color", text="Background Color")
        col.separator()
        col.prop(props, "display_mode", text="")
        col.prop(props, "opacity", slider=True)

        layout.separator()

        # ── Start / Stop ──────────────────────────────────────────────────────
        if not props.is_running:
            row = layout.row()
            row.scale_y = 1.8
            row.operator("nerf.start_live_render", icon="PLAY",
                         text="Start NeRF View")
        else:
            row = layout.row()
            row.scale_y = 1.4
            row.operator("nerf.stop_live_render", icon="SNAP_FACE",
                         text="Stop")

            if _active_state:
                with _active_state._lock:
                    status = _active_state.status
                    error  = _active_state.error
                    fps    = _active_state.fps

                box = layout.box()
                if error:
                    col = box.column(align=True)
                    col.alert = True
                    col.label(text=status, icon="ERROR")
                    for i in range(0, len(error), 42):
                        col.label(text=error[i:i+42])
                else:
                    col = box.column(align=True)
                    col.label(text=status, icon="CHECKMARK")
                    if fps > 0:
                        col.label(text=f"FPS: {fps:.1f}", icon="TIME")

        # ── Tip ───────────────────────────────────────────────────────────────
        box = layout.box()
        box.label(text="Scale Note:", icon="INFO")
        col = box.column(align=True)
        col.scale_y = 0.75
        col.label(text="model_manager uses: --scale 0.7")
        col.label(text="-> Set '--scale' to 0.7 here")
        col.label(text="Rotate view using Middle Mouse")
        # Add is_running at the end
    is_running: bpy.props.BoolProperty(name="Is Running", default=False)


# ──────────────────────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────────────────────

CLASSES = [
    NeRFProperties,
    NERF_OT_StartLiveRender,
    NERF_OT_StopLiveRender,
    VIEW3D_PT_NeRFLiveRender,
]


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.nerf_props = bpy.props.PointerProperty(type=NeRFProperties)
    print("[NeRF Addon] Registered.")


def unregister():
    global _active_state, _draw_handle
    if _active_state:
        _active_state.running = False
    if _draw_handle:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, "WINDOW")
        except Exception:
            pass
        _draw_handle = None

    del bpy.types.Scene.nerf_props
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    print("[NeRF Addon] Unregistered.")


if __name__ == "__main__":
    register()
