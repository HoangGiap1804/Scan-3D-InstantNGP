"""
blender_nerf_addon.py  --  NeRF Live Render Add-on for Blender
==============================================================
NeRF objects appear behind Blender 3D objects using GPU depth test.

HOW DEPTH COMPOSITING WORKS (Blender 5.0 compatible):
  POST_VIEW callback draws NeRF while scene depth buffer is still active.
  The fragment shader writes gl_FragDepth = converted NeRF depth.
  GPU depth test (LESS_EQUAL) automatically discards NeRF fragments that
  are further than existing Blender geometry.  No depth buffer *reading*
  needed - the GPU handles everything.

Installation:
    Edit > Preferences > Add-ons > Install...
    Select this file -> Enable "NeRF Live Render"

Blender requirement: 3.0+
"""

bl_info = {
    "name": "NeRF Live Render",
    "author": "NeRF Project",
    "version": (3, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > N-Panel > NeRF",
    "description": "Live NeRF render in Blender viewport with GPU depth compositing",
    "category": "Render",
}

import bpy
import gpu
import socket
import struct
import json
import io
import threading
import time
import numpy as np

from gpu_extras.batch import batch_for_shader


# =============================================================================
# GLSL Shaders
# =============================================================================

# ── Simple overlay (no depth) — used in POST_PIXEL ───────────────────────────
_SIMPLE_VERT = """
    uniform vec2 viewport_size;
    in vec2 pos;
    in vec2 texCoord;
    out vec2 vUV;
    void main() {
        vec2 ndc = 2.0 * (pos / viewport_size) - 1.0;
        gl_Position = vec4(ndc, 0.0, 1.0);
        vUV = texCoord;
    }
"""
_SIMPLE_FRAG = """
    uniform sampler2D image;
    uniform float opacity;
    in vec2 vUV;
    out vec4 fragColor;
    void main() {
        vec4 c = texture(image, vUV);
        if (c.a < 0.004) discard;
        fragColor = vec4(c.rgb, c.a * opacity);
    }
"""

# ── Depth-write shader — used in POST_VIEW ───────────────────────────────────
# Writes gl_FragDepth so the GPU depth test automatically discards NeRF
# fragments that are further away than existing Blender geometry.
# No depth *reading* required — works on every Blender/GPU combination.
_DEPTH_VERT = """
    uniform vec2 viewport_size;
    in vec2 pos;
    in vec2 texCoord;
    out vec2 vUV;
    void main() {
        vec2 ndc = 2.0 * (pos / viewport_size) - 1.0;
        // z=0.5 is a placeholder; gl_FragDepth overrides it per-pixel.
        gl_Position = vec4(ndc, 0.5, 1.0);
        vUV = texCoord;
    }
"""
_DEPTH_FRAG = """
    uniform sampler2D nerf_color;
    uniform sampler2D nerf_depth_tex;
    uniform float opacity;
    uniform float nerf_near;   // NeRF near in Blender units (= min_near / scale)
    uniform float nerf_far;    // NeRF far  in Blender units (= 2*bound / scale)
    uniform float bl_near;     // Blender clip_start
    uniform float bl_far;      // Blender clip_end
    in vec2 vUV;
    out vec4 fragColor;

    float nerf_to_gl_depth(float nd) {
        // nd: NeRF normalized depth [0,1]
        // Step 1: -> camera-space distance (Blender units, positive)
        float z = max(nerf_near + nd * (nerf_far - nerf_near), 0.0001);
        // Step 2: perspective -> OpenGL NDC depth [0,1]
        //   z_ndc = (f+n)/(f-n) - 2fn / ((f-n) * z_cam)
        //   depth_buf = 0.5 * z_ndc + 0.5
        float fn   = bl_far + bl_near;
        float fmn  = bl_far - bl_near;
        float z_ndc = fn / fmn - (2.0 * bl_far * bl_near) / (fmn * z);
        return clamp(0.5 * z_ndc + 0.5, 0.0, 1.0);
    }

    void main() {
        vec4 c = texture(nerf_color, vUV);
        if (c.a < 0.004) discard;

        float nd = texture(nerf_depth_tex, vUV).r;
        // Write converted depth -> GPU depth test (LESS_EQUAL) automatically
        // discards this fragment if NeRF is behind a Blender object.
        gl_FragDepth = nerf_to_gl_depth(nd);

        fragColor = vec4(c.rgb, c.a * opacity);
    }
"""


# =============================================================================
# Constants
# =============================================================================

TIMER_INTERVAL  = 0.033
REQUEST_TIMEOUT = 5.0
MAX_RENDER_DIM  = 1920


# =============================================================================
# TCP helpers
# =============================================================================

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
    hdr = _recv_exactly(sock, 4)
    if hdr is None:
        return None
    return _recv_exactly(sock, struct.unpack(">I", hdr)[0])

def send_framed(sock, data):
    sock.sendall(struct.pack(">I", len(data)) + data)


# =============================================================================
# Camera helpers (matches nerf_matrix_to_ngp in provider.py)
# =============================================================================

def blender_to_nerf_matrix(pose_blender, scale, offset):
    p = pose_blender
    return np.array([
        [ p[1,0], -p[1,1], -p[1,2],  p[1,3]*scale + offset[0]],
        [ p[2,0], -p[2,1], -p[2,2],  p[2,3]*scale + offset[1]],
        [ p[0,0], -p[0,1], -p[0,2],  p[0,3]*scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)

def get_blender_camera_pose(region_3d, scale=1.0, offset=None):
    if offset is None:
        offset = [0.0, 0.0, 0.0]
    view = np.array(region_3d.view_matrix, dtype=np.float32).reshape(4, 4)
    return blender_to_nerf_matrix(np.linalg.inv(view), scale=scale, offset=offset)

def get_blender_intrinsics(region_3d, W, H):
    P  = np.array(region_3d.window_matrix, dtype=np.float32)
    fx = (W / 2.0) * P[0, 0]
    fy = (H / 2.0) * P[1, 1]
    return np.array([fx, fy, W / 2.0, H / 2.0], dtype=np.float32)


# =============================================================================
# GPU texture helpers
# =============================================================================

def numpy_to_gpu_texture(image):
    """[H,W,3] or [H,W,4] float32 -> GPUTexture (flips Y)."""
    H, W = image.shape[:2]
    img  = image[::-1, :, :].copy()
    if image.shape[2] == 4:
        rgba = img.astype(np.float32)
    else:
        rgba = np.ones((H, W, 4), dtype=np.float32)
        rgba[:, :, :3] = img
    buf = gpu.types.Buffer("FLOAT", H * W * 4, rgba.flatten())
    return gpu.types.GPUTexture((W, H), format="RGBA32F", data=buf)

def numpy_depth_to_gpu_texture(depth_arr):
    """[H,W] float32 -> GPUTexture (R32F, no Y-flip needed: already flipped with color)."""
    H, W = depth_arr.shape
    # Flip Y to match the color texture (which was flipped in numpy_to_gpu_texture)
    flipped = depth_arr[::-1, :].copy()
    buf = gpu.types.Buffer("FLOAT", H * W, flipped.flatten())
    return gpu.types.GPUTexture((W, H), format="R32F", data=buf)


# =============================================================================
# PNG decode
# =============================================================================

def decode_png_to_numpy(png_bytes):
    """PNG bytes -> [H,W,3] or [H,W,4] float32 [0,1]."""
    try:
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(png_bytes))
        mode = 'RGBA' if img.mode == 'RGBA' else 'RGB'
        return np.array(img.convert(mode), dtype=np.float32) / 255.0
    except ImportError:
        import cv2
        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError("cv2.imdecode failed")
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        elif img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0


# =============================================================================
# Shared State
# =============================================================================

class SharedState:
    def __init__(self):
        self._lock    = threading.Lock()
        self.running  = False

        self.latest_pose        = None
        self.latest_intrinsics  = None
        self.render_W           = 512
        self.render_H           = 512
        self.bg_color           = [1.0, 1.0, 1.0]
        self.transparent        = True
        self.include_depth      = False

        self.latest_image   = None
        self.latest_depth   = None
        self.image_updated  = False
        self.image_w        = 512
        self.image_h        = 512

        self.fps    = 0.0
        self.status = "Disconnected"
        self.error  = ""

    def set_camera(self, pose, intrinsics, W, H, bg,
                   transparent=True, include_depth=False):
        with self._lock:
            self.latest_pose       = pose
            self.latest_intrinsics = intrinsics
            self.render_W          = W
            self.render_H          = H
            self.bg_color          = bg
            self.transparent       = transparent
            self.include_depth     = include_depth

    def get_camera(self):
        with self._lock:
            return (self.latest_pose, self.latest_intrinsics,
                    self.render_W, self.render_H, self.bg_color,
                    self.transparent, self.include_depth)

    def set_render(self, image, depth=None):
        with self._lock:
            self.latest_image  = image
            self.latest_depth  = depth
            self.image_updated = True
            self.image_h, self.image_w = image.shape[:2]

    def get_render_if_new(self):
        with self._lock:
            if self.image_updated:
                self.image_updated = False
                return self.latest_image, self.latest_depth, self.image_w, self.image_h
            return None, None, None, None

    def get_image_size(self):
        with self._lock:
            return self.image_w, self.image_h

    def set_status(self, status, error="", fps=0.0):
        with self._lock:
            self.status = status
            self.error  = error
            self.fps    = fps


# =============================================================================
# Network Thread
# =============================================================================

def network_thread_fn(host, port, state):
    sock = None
    try:
        state.set_status("Connecting...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(REQUEST_TIMEOUT)
        sock.connect((host, port))
        state.set_status(f"Connected to {host}:{port}")
        print(f"[NeRF Addon] Connected to {host}:{port}")

        t_prev = time.perf_counter()
        while state.running:
            pose, intrinsics, W, H, bg, transparent, include_depth = state.get_camera()
            if pose is None:
                time.sleep(0.01)
                continue

            req = {
                "pose":          pose.tolist(),
                "intrinsics":    intrinsics.tolist(),
                "W": W, "H": H,
                "bg_color":      bg,
                "transparent":   transparent,
                "include_depth": include_depth,
            }
            send_framed(sock, json.dumps(req).encode("utf-8"))

            png_bytes = recv_framed(sock)
            if png_bytes is None:
                state.set_status("Error: connection lost", error="Server closed")
                break
            image = decode_png_to_numpy(png_bytes)

            depth = None
            if include_depth:
                depth_bytes = recv_framed(sock)
                if depth_bytes is not None:
                    depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape(H, W)

            state.set_render(image, depth)
            t_now  = time.perf_counter()
            fps    = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now
            state.set_status(f"Rendering | {W}x{H}", fps=fps)

    except ConnectionRefusedError:
        state.set_status("Error: refused",
                         error=f"Cannot connect to {host}:{port}")
    except socket.timeout:
        state.set_status("Error: timeout", error="Server did not respond")
    except Exception as exc:
        state.set_status(f"Error: {type(exc).__name__}", error=str(exc))
        import traceback; traceback.print_exc()
    finally:
        if sock:
            try: sock.close()
            except Exception: pass
        state.running = False
        print("[NeRF Addon] Network thread stopped.")


# =============================================================================
# Display helpers
# =============================================================================

def _compute_draw_rect(rw, rh, iw, ih, mode):
    if mode == 'FILL':
        return 0, 0, rw, rh
    elif mode == 'FIT':
        s  = min(rw / iw, rh / ih)
        dw, dh = int(iw * s), int(ih * s)
        return (rw - dw) // 2, (rh - dh) // 2, dw, dh
    else:  # CORNER
        dw, dh = min(iw, rw), min(ih, rh)
        return rw - dw, 0, dw, dh

def _build_batch(shader, x, y, w, h):
    verts = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
    uvs   = [(0, 0), (1,   0), (1,   1   ), (0, 1   )]
    return batch_for_shader(shader, "TRIS",
                            {"pos": verts, "texCoord": uvs},
                            indices=[(0,1,2),(0,2,3)])


# =============================================================================
# Global render state
# =============================================================================

_active_state   = None
_handle_pixel   = None   # POST_PIXEL: simple color overlay
_handle_view    = None   # POST_VIEW:  depth-aware drawing

_gpu_color_tex  = None   # NeRF color   (RGBA32F)
_gpu_depth_tex  = None   # NeRF depth   (R32F)

_simple_shader  = None
_depth_shader   = None

_simple_batch   = None
_depth_batch    = None
_simple_key     = None
_depth_key      = None

_is_premult     = False


# =============================================================================
# Draw Callbacks
# =============================================================================

def _get_space_clips(context):
    """Return (near, far) from the active VIEW_3D space."""
    try:
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                s = area.spaces.active
                return s.clip_start, s.clip_end
    except Exception:
        pass
    return 0.1, 1000.0


def draw_nerf_simple(context):
    """
    POST_PIXEL callback: plain color overlay, no depth compositing.
    Skipped when depth_compositing is enabled (POST_VIEW handles that).
    """
    global _active_state, _gpu_color_tex
    global _simple_shader, _simple_batch, _simple_key, _is_premult

    if _active_state is None or _gpu_color_tex is None:
        return
    if not hasattr(context.scene, 'nerf_props'):
        return
    props = context.scene.nerf_props
    if props.depth_compositing:
        return   # Handled by POST_VIEW callback

    region = context.region
    rw, rh = region.width, region.height
    iw, ih = _active_state.get_image_size()
    mode   = props.display_mode
    key    = (rw, rh, iw, ih, mode)

    if _simple_shader is None:
        try:
            _simple_shader = gpu.types.GPUShader(_SIMPLE_VERT, _SIMPLE_FRAG)
        except Exception as e:
            print(f"[NeRF] Simple shader error: {e}")
            return

    if key != _simple_key or _simple_batch is None:
        x, y, dw, dh = _compute_draw_rect(rw, rh, iw, ih, mode)
        _simple_batch = _build_batch(_simple_shader, x, y, dw, dh)
        _simple_key   = key

    blend = "ALPHA_PREMULT" if _is_premult else "ALPHA"
    gpu.state.blend_set(blend)
    _simple_shader.bind()
    _simple_shader.uniform_sampler("image",         _gpu_color_tex)
    _simple_shader.uniform_float("opacity",         props.opacity)
    _simple_shader.uniform_float("viewport_size",   (rw, rh))
    if _simple_batch:
        _simple_batch.draw(_simple_shader)
    gpu.state.blend_set("NONE")


def draw_nerf_depth(context):
    """
    POST_VIEW callback: depth-aware compositing using GPU depth test.

    How it works:
      - Fires after Blender has rendered its 3D scene into the viewport FBO.
      - The depth buffer now contains Blender scene depth values.
      - We draw the NeRF quad and write gl_FragDepth = converted NeRF depth.
      - GPU depth test (LESS_EQUAL) automatically discards NeRF pixels that
        are further away than existing Blender geometry.
      - No depth buffer *reading* required — 100% GPU-side, works on
        Blender 3.0 – 5.x regardless of bgl availability.
    """
    global _active_state, _gpu_color_tex, _gpu_depth_tex
    global _depth_shader, _depth_batch, _depth_key, _is_premult

    if _active_state is None or _gpu_color_tex is None or _gpu_depth_tex is None:
        return
    if not hasattr(context.scene, 'nerf_props'):
        return
    props = context.scene.nerf_props
    if not props.depth_compositing or not props.is_running:
        return

    region = context.region
    rw, rh = region.width, region.height
    iw, ih = _active_state.get_image_size()
    mode   = props.display_mode
    key    = (rw, rh, iw, ih, mode)

    if _depth_shader is None:
        try:
            _depth_shader = gpu.types.GPUShader(_DEPTH_VERT, _DEPTH_FRAG)
        except Exception as e:
            print(f"[NeRF] Depth shader compile error: {e}")
            return

    if key != _depth_key or _depth_batch is None:
        x, y, dw, dh = _compute_draw_rect(rw, rh, iw, ih, mode)
        _depth_batch  = _build_batch(_depth_shader, x, y, dw, dh)
        _depth_key    = key

    bl_near, bl_far = _get_space_clips(context)

    blend = "ALPHA_PREMULT" if _is_premult else "ALPHA"
    gpu.state.blend_set(blend)
    # LESS_EQUAL: pass if NeRF depth <= existing depth (NeRF in front or same)
    gpu.state.depth_test_set("LESS_EQUAL")
    # Don't overwrite Blender's scene depth so subsequent draws stay correct
    gpu.state.depth_mask_set(False)

    _depth_shader.bind()
    _depth_shader.uniform_sampler("nerf_color",    _gpu_color_tex)
    _depth_shader.uniform_sampler("nerf_depth_tex", _gpu_depth_tex)
    _depth_shader.uniform_float("opacity",         props.opacity)
    _depth_shader.uniform_float("nerf_near",       props.nerf_depth_near)
    _depth_shader.uniform_float("nerf_far",        props.nerf_depth_far)
    _depth_shader.uniform_float("bl_near",         bl_near)
    _depth_shader.uniform_float("bl_far",          bl_far)
    _depth_shader.uniform_float("viewport_size",   (rw, rh))

    if _depth_batch:
        _depth_batch.draw(_depth_shader)

    # Restore state
    gpu.state.depth_test_set("NONE")
    gpu.state.depth_mask_set(True)
    gpu.state.blend_set("NONE")


# =============================================================================
# Operators
# =============================================================================

class NERF_OT_StartLiveRender(bpy.types.Operator):
    """Start NeRF Server connection and display render in viewport"""
    bl_idname  = "nerf.start_live_render"
    bl_label   = "Start NeRF View"
    bl_options = {"REGISTER"}

    _timer = None

    def modal(self, context, event):
        global _active_state, _gpu_color_tex, _gpu_depth_tex
        global _simple_batch, _depth_batch, _simple_key, _depth_key, _is_premult

        if _active_state is None or not _active_state.running:
            self.cancel(context)
            return {"CANCELLED"}

        if event.type == "TIMER":
            props      = context.scene.nerf_props
            region_3d  = None
            region_win = None

            for area in context.screen.areas:
                if area.type == "VIEW_3D":
                    for r in area.regions:
                        if r.type == "WINDOW":
                            s = area.spaces.active
                            if hasattr(s, "region_3d"):
                                region_3d  = s.region_3d
                                region_win = r
                                break
                    if region_3d:
                        break

            if region_3d is None:
                return {"PASS_THROUGH"}

            if props.use_viewport_size:
                ds = max(0.1, min(1.0, props.viewport_downscale))
                render_W = max(64, int(region_win.width  * ds))
                render_H = max(64, int(region_win.height * ds))
            else:
                render_W = min(props.render_width,  MAX_RENDER_DIM)
                render_H = min(props.render_height, MAX_RENDER_DIM)

            scale  = props.nerf_scale
            offset = [props.nerf_offset_x, props.nerf_offset_y, props.nerf_offset_z]
            pose   = get_blender_camera_pose(region_3d, scale=scale, offset=offset)
            intr   = get_blender_intrinsics(region_3d, render_W, render_H)

            bg_r, bg_g, bg_b = props.bg_color
            _active_state.set_camera(
                pose, intr, render_W, render_H,
                [bg_r, bg_g, bg_b],
                transparent   = props.transparent,
                include_depth = props.depth_compositing,
            )

            img, dep, iw, ih = _active_state.get_render_if_new()
            if img is not None:
                _gpu_color_tex = numpy_to_gpu_texture(img)
                _is_premult    = (img.ndim == 3 and img.shape[2] == 4)
                _gpu_depth_tex = (numpy_depth_to_gpu_texture(dep)
                                  if dep is not None else None)
                _simple_batch  = None
                _depth_batch   = None
                _simple_key    = None
                _depth_key     = None
                for area in context.screen.areas:
                    if area.type == "VIEW_3D":
                        area.tag_redraw()

        return {"PASS_THROUGH"}

    def invoke(self, context, event):
        global _active_state, _handle_pixel, _handle_view
        global _gpu_color_tex, _gpu_depth_tex
        global _simple_shader, _depth_shader
        global _simple_batch, _depth_batch

        props = context.scene.nerf_props

        _active_state  = SharedState()
        _active_state.running = True
        _gpu_color_tex = None
        _gpu_depth_tex = None
        _simple_shader = None
        _depth_shader  = None
        _simple_batch  = None
        _depth_batch   = None

        # POST_PIXEL: simple overlay (used when depth_compositing=False)
        _handle_pixel = bpy.types.SpaceView3D.draw_handler_add(
            draw_nerf_simple, (context,), "WINDOW", "POST_PIXEL"
        )
        # POST_VIEW: depth-aware drawing (used when depth_compositing=True)
        # Fires BEFORE overlays so scene depth buffer is still intact.
        _handle_view = bpy.types.SpaceView3D.draw_handler_add(
            draw_nerf_depth, (context,), "WINDOW", "POST_VIEW"
        )

        threading.Thread(
            target=network_thread_fn,
            args=(props.host, props.port, _active_state),
            daemon=True,
        ).start()

        self._timer = context.window_manager.event_timer_add(
            TIMER_INTERVAL, window=context.window
        )
        context.window_manager.modal_handler_add(self)
        props.is_running = True
        print(f"[NeRF Addon] Connecting to {props.host}:{props.port} ...")
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        global _active_state, _handle_pixel, _handle_view
        global _gpu_color_tex, _gpu_depth_tex

        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        for handle_attr in ('_handle_pixel', '_handle_view'):
            h = globals().get(handle_attr)
            if h:
                try:
                    bpy.types.SpaceView3D.draw_handler_remove(h, "WINDOW")
                except Exception:
                    pass
                globals()[handle_attr] = None
        if _active_state:
            _active_state.running = False
            _active_state = None
        _gpu_color_tex = None
        _gpu_depth_tex = None

        if hasattr(context, 'scene') and hasattr(context.scene, 'nerf_props'):
            context.scene.nerf_props.is_running = False
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


class NERF_OT_StopLiveRender(bpy.types.Operator):
    """Stop NeRF connection and remove overlay"""
    bl_idname  = "nerf.stop_live_render"
    bl_label   = "Stop"
    bl_options = {"REGISTER"}

    def execute(self, context):
        global _active_state, _handle_pixel, _handle_view
        global _gpu_color_tex, _gpu_depth_tex

        if _active_state:
            _active_state.running = False
        for handle_attr in ('_handle_pixel', '_handle_view'):
            h = globals().get(handle_attr)
            if h:
                try:
                    bpy.types.SpaceView3D.draw_handler_remove(h, "WINDOW")
                except Exception:
                    pass
                globals()[handle_attr] = None
        _gpu_color_tex = None
        _gpu_depth_tex = None

        context.scene.nerf_props.is_running = False
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
        self.report({"INFO"}, "NeRF Live Render stopped")
        return {"FINISHED"}


# =============================================================================
# Properties
# =============================================================================

class NeRFProperties(bpy.types.PropertyGroup):
    host: bpy.props.StringProperty(name="Host", default="127.0.0.1")
    port: bpy.props.IntProperty(name="Port", default=6789, min=1024, max=65535)

    use_viewport_size: bpy.props.BoolProperty(
        name="Auto Viewport Size",
        description="Use viewport dimensions as render resolution",
        default=True,
    )
    viewport_downscale: bpy.props.FloatProperty(
        name="Downscale",
        description="1.0 = full viewport resolution, 0.5 = half (faster)",
        default=0.5, min=0.1, max=1.0, subtype="FACTOR",
    )
    render_width:  bpy.props.IntProperty(name="W", default=512, min=64, max=MAX_RENDER_DIM)
    render_height: bpy.props.IntProperty(name="H", default=512, min=64, max=MAX_RENDER_DIM)

    nerf_scale: bpy.props.FloatProperty(
        name="--scale",
        description="Must match --scale when running nerf_server.py",
        default=0.7, min=0.001, max=10.0, precision=4,
    )
    nerf_offset_x: bpy.props.FloatProperty(name="Offset X", default=0.0, precision=3)
    nerf_offset_y: bpy.props.FloatProperty(name="Offset Y", default=0.0, precision=3)
    nerf_offset_z: bpy.props.FloatProperty(name="Offset Z", default=0.0, precision=3)

    transparent: bpy.props.BoolProperty(
        name="Transparent BG",
        description="Render with transparent background (RGBA)",
        default=True,
    )
    bg_color: bpy.props.FloatVectorProperty(
        name="BG Color", default=(1.0, 1.0, 1.0),
        min=0.0, max=1.0, subtype="COLOR",
    )

    display_mode: bpy.props.EnumProperty(
        name="Display Mode",
        items=[
            ('FILL',   "Fill",   "Fill viewport"),
            ('FIT',    "Fit",    "Preserve aspect ratio, centred"),
            ('CORNER', "Corner", "Bottom-right corner"),
        ],
        default='FILL',
    )
    opacity: bpy.props.FloatProperty(
        name="Opacity", default=1.0, min=0.0, max=1.0, subtype="FACTOR",
    )

    # ── Depth Compositing ────────────────────────────────────────────────────
    depth_compositing: bpy.props.BoolProperty(
        name="Depth Compositing",
        description=(
            "NeRF objects appear behind Blender 3D objects.\n"
            "Uses GPU depth test — works on Blender 3.0 – 5.x without any extra modules."
        ),
        default=False,
    )
    nerf_depth_near: bpy.props.FloatProperty(
        name="NeRF Near (Blender units)",
        description="Camera distance at NeRF depth=0.  Formula: min_near / scale  (e.g. 0.2/0.7 ≈ 0.29)",
        default=0.29, min=0.001, max=100.0, precision=3,
    )
    nerf_depth_far: bpy.props.FloatProperty(
        name="NeRF Far (Blender units)",
        description="Camera distance at NeRF depth=1.  Formula: 2×bound / scale  (e.g. 2×1/0.7 ≈ 2.86)",
        default=2.86, min=0.1, max=1000.0, precision=3,
    )

    is_running: bpy.props.BoolProperty(name="Is Running", default=False)


# =============================================================================
# Panel
# =============================================================================

class VIEW3D_PT_NeRFLiveRender(bpy.types.Panel):
    bl_label       = "NeRF Live Render"
    bl_idname      = "VIEW3D_PT_nerf_live_render"
    bl_space_type  = "VIEW_3D"
    bl_region_type = "UI"
    bl_category    = "NeRF"

    def draw(self, context):
        layout = self.layout
        props  = context.scene.nerf_props

        # Connection
        box = layout.box()
        box.label(text="Server", icon="NETWORK_DRIVE")
        col = box.column(align=True)
        col.prop(props, "host")
        col.prop(props, "port")

        # Render size
        box = layout.box()
        box.label(text="Render Size", icon="IMAGE_DATA")
        col = box.column(align=True)
        col.prop(props, "use_viewport_size")
        if props.use_viewport_size:
            col.prop(props, "viewport_downscale", slider=True)
        else:
            row = col.row(align=True)
            row.prop(props, "render_width")
            row.prop(props, "render_height")

        # Coordinate alignment
        box = layout.box()
        box.label(text="Coordinate Alignment", icon="ORIENTATION_GLOBAL")
        col = box.column(align=True)
        col.prop(props, "nerf_scale")
        row = col.row(align=True)
        row.label(text="Offset:")
        row.prop(props, "nerf_offset_x", text="X")
        row.prop(props, "nerf_offset_y", text="Y")
        row.prop(props, "nerf_offset_z", text="Z")

        # Background + Display
        box = layout.box()
        box.label(text="Background & Display", icon="RENDERLAYERS")
        col = box.column(align=True)
        col.prop(props, "transparent")
        if not props.transparent:
            col.prop(props, "bg_color")
        col.separator()
        col.prop(props, "display_mode", text="")
        col.prop(props, "opacity", slider=True)

        # Depth compositing
        box = layout.box()
        box.prop(props, "depth_compositing", icon="IMAGE_ZDEPTH")
        if props.depth_compositing:
            col = box.column(align=True)
            col.prop(props, "nerf_depth_near", text="Near (Bl units)")
            col.prop(props, "nerf_depth_far",  text="Far  (Bl units)")
            col.separator()
            sub = col.column(align=True)
            sub.scale_y = 0.75
            sub.label(text="Near = min_near / scale", icon="INFO")
            sub.label(text="Far  = 2 x bound / scale")
            sub.label(text="e.g. scale=0.7 bound=1:")
            sub.label(text="  Near=0.29  Far=2.86")

        layout.separator()

        # Start / Stop
        if not props.is_running:
            row = layout.row()
            row.scale_y = 1.8
            row.operator("nerf.start_live_render", icon="PLAY", text="Start NeRF View")
        else:
            row = layout.row()
            row.scale_y = 1.4
            row.operator("nerf.stop_live_render", icon="PAUSE", text="Stop")

            if _active_state:
                with _active_state._lock:
                    status = _active_state.status
                    error  = _active_state.error
                    fps    = _active_state.fps

                box = layout.box()
                col = box.column(align=True)
                if error:
                    col.alert = True
                    col.label(text=status, icon="ERROR")
                    for i in range(0, len(error), 42):
                        col.label(text=error[i:i+42])
                else:
                    col.label(text=status, icon="CHECKMARK")
                    if fps > 0:
                        col.label(text=f"FPS: {fps:.1f}", icon="TIME")
                    if props.depth_compositing:
                        col.label(text="Depth: ON", icon="IMAGE_ZDEPTH")


# =============================================================================
# Registration
# =============================================================================

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
    print("[NeRF Addon] v3.0 registered — GPU depth compositing enabled.")

def unregister():
    global _active_state, _handle_pixel, _handle_view
    if _active_state:
        _active_state.running = False
    for h in [_handle_pixel, _handle_view]:
        if h:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(h, "WINDOW")
            except Exception:
                pass
    _handle_pixel = None
    _handle_view  = None
    del bpy.types.Scene.nerf_props
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    print("[NeRF Addon] Unregistered.")

if __name__ == "__main__":
    register()
