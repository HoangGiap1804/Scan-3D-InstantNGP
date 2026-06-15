import dearpygui.dearpygui as dpg
import subprocess
import shlex
import threading
import json
import os
import atexit

HISTORY_FILE = "recent_files.json"
TEXTURE_CACHE = {} 

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving history: {e}")

history_list = load_history()

def update_and_render_history(file_path):
    if file_path in history_list:
        history_list.remove(file_path)
    history_list.insert(0, file_path) 
    
    if len(history_list) > 12:
        history_list.pop()
        
    save_history(history_list)
    render_history_grid()

def file_selected_cb(sender, app_data):
    file_path = app_data.get('file_path_name', "")
    if file_path:
        update_and_render_history(file_path)

def get_first_image_for_dataset(json_path):
    dir_path = os.path.dirname(json_path)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'frames' in data and len(data['frames']) > 0:
            frame_path = data['frames'][0].get('file_path', '')
            if frame_path:
                if frame_path.startswith('./'):
                    frame_path = frame_path[2:]
                for ext in ['', '.png', '.jpg', '.jpeg']:
                    test_path = os.path.join(dir_path, frame_path + ext)
                    if os.path.isfile(test_path):
                        return test_path
    except Exception:
        pass
        
    try:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    return os.path.join(root, file)
            if root != dir_path:
                break
    except Exception:
        pass
        
    return None

def render_history_grid():
    global active_server_file_path, active_server_share_btn, active_server_stop_btn
    dpg.delete_item("history_grid", children_only=True)
    active_server_share_btn = None
    active_server_stop_btn = None
    
    if not history_list:
        dpg.add_text("No recent files yet.", parent="history_grid", color=[150, 150, 150])
        return

    table_id = dpg.add_table(header_row=False, parent="history_grid", borders_innerH=False, borders_innerV=False, borders_outerH=False, borders_outerV=False)
    dpg.add_table_column(parent=table_id)
    dpg.add_table_column(parent=table_id)
    dpg.add_table_column(parent=table_id)
    
    row_id = None
    for i, file_path in enumerate(history_list):
        if i % 3 == 0:
            row_id = dpg.add_table_row(parent=table_id)
            
        display_name = os.path.basename(os.path.dirname(file_path))
        if not display_name:
            display_name = os.path.basename(file_path)
            
        img_path = get_first_image_for_dataset(file_path)
        tex_id = None
        
        if img_path:
            if img_path in TEXTURE_CACHE:
                tex_id = TEXTURE_CACHE[img_path]
            else:
                try:
                    w, h, c, data = dpg.load_image(img_path)
                    tex_id = f"tex_{len(TEXTURE_CACHE)}"
                    dpg.add_static_texture(width=w, height=h, default_value=data, parent="global_texture_registry", tag=tex_id)
                    TEXTURE_CACHE[img_path] = tex_id
                except Exception as e:
                    tex_id = None
                    
        # Tạo thiết kế dạng thẻ (Card) tuyệt đẹp bằng child_window
        with dpg.child_window(parent=row_id, width=240, height=318, border=True, no_scrollbar=True) as card:
            # Hàng trên cùng: nút xóa góc phải
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=-1)
                btn_del = dpg.add_button(label=" X ", width=28, height=22,
                                         callback=delete_item_cb, user_data=file_path)
                dpg.bind_item_theme(btn_del, "delete_btn_theme")

            # Ảnh dataset
            if tex_id:
                dpg.add_image(tex_id, width=214, height=190)
            else:
                with dpg.group(horizontal=False):
                    dpg.add_spacer(height=70)
                    dpg.add_text("      No Preview", color=[190, 195, 210])
                    dpg.add_spacer(height=70)

            # Tên dataset
            txt_btn = dpg.add_button(label=display_name, width=214, height=30)

            # Nút hành động
            with dpg.group(horizontal=True):
                btn_train = dpg.add_button(label="Train", width=105, height=30, callback=open_training_popup_cb, user_data=file_path)

                show_share = (file_path != active_server_file_path)
                btn_server = dpg.add_button(label="Share", width=105, height=30, callback=run_server_cb, user_data=file_path, show=show_share)
                btn_stop   = dpg.add_button(label="Stop",  width=105, height=30, callback=stop_server_cb, user_data=file_path, show=not show_share)

                if not show_share:
                    active_server_share_btn = btn_server
                    active_server_stop_btn  = btn_stop

            dpg.bind_item_theme(card,       "card_theme")
            dpg.bind_item_theme(txt_btn,    "text_btn_theme")
            dpg.bind_item_theme(btn_train,  "action_btn_theme")
            dpg.bind_item_theme(btn_server, "share_btn_theme")
            dpg.bind_item_theme(btn_stop,   "stop_btn_theme")

active_server_process = None
active_server_file_path = None
active_server_share_btn = None
active_server_stop_btn = None
running_processes = []

def cleanup_all_processes():
    for p in running_processes:
        try:
            p.terminate()
            p.kill()
        except:
            pass

atexit.register(cleanup_all_processes)

def stop_server_cb(sender=None, app_data=None, user_data=None):
    global active_server_process, active_server_file_path, active_server_share_btn, active_server_stop_btn
    if active_server_process:
        try:
            active_server_process.terminate()
        except:
            pass
        if active_server_process in running_processes:
            running_processes.remove(active_server_process)
        active_server_process = None
        active_server_file_path = None
        
        dpg.set_value("server_status_text", "Server: Stopped")
        dpg.configure_item("server_status_text", color=[200, 80, 80])
        
        if active_server_share_btn and active_server_stop_btn:
            try:
                dpg.configure_item(active_server_share_btn, show=True)
                dpg.configure_item(active_server_stop_btn, show=False)
            except:
                pass

def monitor_server_process(proc):
    global active_server_process, active_server_file_path, active_server_share_btn, active_server_stop_btn
    proc.wait()
    if proc in running_processes:
        running_processes.remove(proc)
    if active_server_process == proc:
        active_server_process = None
        active_server_file_path = None
        
        dpg.set_value("server_status_text", "Server: Stopped (Exited)")
        dpg.configure_item("server_status_text", color=[200, 80, 80])
        
        if active_server_share_btn and active_server_stop_btn:
            try:
                dpg.configure_item(active_server_share_btn, show=True)
                dpg.configure_item(active_server_stop_btn, show=False)
            except:
                pass

def delete_item_cb(sender=None, app_data=None, user_data=None):
    """Xóa một dataset khỏi danh sách lịch sử."""
    file_path = user_data
    if file_path in history_list:
        history_list.remove(file_path)
        save_history(history_list)
    # Nếu đang chạy server cho item này thì dừng
    if file_path == active_server_file_path:
        stop_server_cb()
    render_history_grid()

def launch_command(cmd, status_msg):
    dpg.set_value("status_text", f"Status: {status_msg}")
    def execute():
        try:
            args = shlex.split(cmd)
            proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            running_processes.append(proc)
            proc.wait()
            if proc in running_processes:
                running_processes.remove(proc)
            dpg.set_value("status_text", "Status: Command executed successfully!")
        except Exception as e:
            dpg.set_value("status_text", f"Execution error: {str(e)}")
    threading.Thread(target=execute, daemon=True).start()

CONFIG_FILE = "training_config.json"

def load_training_config():
    default_config = {
        "iters": 30000,
        "bound": 1.0,
        "scale": 0.7,
        "dt_gamma": 0.0,
        "max_steps": 1024,
        "density_thresh": 10.0,
        "lr": 0.01,
        "gui": True
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                user_config = json.load(f)
                default_config.update(user_config)
        except Exception:
            pass
    return default_config

def save_training_config(config):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Error saving training config: {e}")

current_training_file = None

def open_training_popup_cb(sender, app_data, user_data):
    global current_training_file
    current_training_file = user_data
    
    config = load_training_config()
    dpg.set_value("input_iters", config["iters"])
    dpg.set_value("input_bound", config["bound"])
    dpg.set_value("input_scale", config["scale"])
    dpg.set_value("input_dt_gamma", config["dt_gamma"])
    dpg.set_value("input_max_steps", config["max_steps"])
    dpg.set_value("input_density_thresh", config["density_thresh"])
    dpg.set_value("input_lr", config["lr"])
    dpg.set_value("input_gui", config["gui"])
    
    viewport_width = dpg.get_viewport_client_width()
    viewport_height = dpg.get_viewport_client_height()
    dpg.configure_item("training_popup", show=True, pos=[max(0, viewport_width//2 - 200), max(0, viewport_height//2 - 200)])

def confirm_training_cb():
    dpg.configure_item("training_popup", show=False)
    config = {
        "iters": dpg.get_value("input_iters"),
        "bound": dpg.get_value("input_bound"),
        "scale": dpg.get_value("input_scale"),
        "dt_gamma": dpg.get_value("input_dt_gamma"),
        "max_steps": dpg.get_value("input_max_steps"),
        "density_thresh": dpg.get_value("input_density_thresh"),
        "lr": dpg.get_value("input_lr"),
        "gui": dpg.get_value("input_gui")
    }
    save_training_config(config)
    
    file_path = current_training_file
    update_and_render_history(file_path)
    if not file_path:
        return
    dir_path = os.path.dirname(file_path).replace('\\', '/')
    new_workspace = os.path.basename(dir_path) or "my_model"
    
    gui_flag = "--gui" if config["gui"] else ""
    
    cmd = f'python main_nerf.py "{dir_path}" --workspace workspaces/{new_workspace} -O --bound {config["bound"]} --scale {config["scale"]} --dt_gamma {config["dt_gamma"]} --max_steps {config["max_steps"]} --density_thresh {config["density_thresh"]} --iters {config["iters"]} --lr {config["lr"]} --color_space linear --error_map {gui_flag}'
    
    launch_command(cmd, f"Launching training for {new_workspace}...")

def cancel_training_cb():
    dpg.configure_item("training_popup", show=False)

def run_server_cb(sender, app_data, user_data):
    global active_server_process, active_server_file_path
    
    if active_server_process:
        stop_server_cb()
        
    active_server_file_path = user_data
    file_path = user_data
    update_and_render_history(file_path)
    if not file_path:
        return
    dir_path = os.path.dirname(file_path).replace('\\', '/')
    new_workspace = os.path.basename(dir_path) or "my_model"
    cmd = f'python nerf_server.py "{dir_path}" --workspace workspaces/{new_workspace} --test -O --bound 1.0 --scale 0.7 --dt_gamma 0'
    
    dpg.set_value("status_text", f"Status: Launching NeRF Server for {new_workspace}...")
    try:
        args = shlex.split(cmd)
        active_server_process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        running_processes.append(active_server_process)
        dpg.set_value("server_status_text", f"Server: Running ({new_workspace})")
        dpg.configure_item("server_status_text", color=[50, 200, 50])
        dpg.set_value("status_text", "Status: Server launched successfully!")
        
        threading.Thread(target=monitor_server_process, args=(active_server_process,), daemon=True).start()
    except Exception as e:
        dpg.set_value("status_text", f"Execution error: {str(e)}")


dpg.create_context()

# ─── Themes ──────────────────────────────────────────────────────────────────

# Card: white with soft shadow border
with dpg.theme(tag="card_theme"):
    with dpg.theme_component(dpg.mvChildWindow):
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (255, 255, 255))
        dpg.add_theme_color(dpg.mvThemeCol_Border,  (218, 222, 235))
        dpg.add_theme_style(dpg.mvStyleVar_ChildRounding,  12)
        dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,  11,  8)
        dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,     4,  5)

# Dataset name label button
with dpg.theme(tag="text_btn_theme"):
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button,        (240, 243, 252))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (228, 232, 248))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  (218, 224, 244))
        dpg.add_theme_color(dpg.mvThemeCol_Text,          ( 52,  72, 155))
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,  7)

# Train – indigo/violet
with dpg.theme(tag="action_btn_theme"):
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button,        ( 92,  75, 192))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (112,  95, 215))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  ( 74,  60, 165))
        dpg.add_theme_color(dpg.mvThemeCol_Text,          (255, 255, 255))
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,  7)

# Share – teal/emerald
with dpg.theme(tag="share_btn_theme"):
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button,        ( 18, 140, 126))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, ( 30, 165, 148))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  ( 12, 115, 103))
        dpg.add_theme_color(dpg.mvThemeCol_Text,          (255, 255, 255))
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,  7)

# Delete X – light danger
with dpg.theme(tag="delete_btn_theme"):
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button,        (254, 242, 242))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (220,  53,  69))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  (185,  35,  50))
        dpg.add_theme_color(dpg.mvThemeCol_Text,          (180,  30,  45))
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,  5)
        dpg.add_theme_style(dpg.mvStyleVar_FramePadding,   2,  1)

# Stop – orange-red
with dpg.theme(tag="stop_btn_theme"):
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button,        (205,  75,  45))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (228,  95,  60))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  (172,  58,  32))
        dpg.add_theme_color(dpg.mvThemeCol_Text,          (255, 255, 255))
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,  7)

# ADD Dataset header button
with dpg.theme(tag="add_btn_theme"):
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button,        ( 92,  75, 192))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (112,  95, 215))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  ( 74,  60, 165))
        dpg.add_theme_color(dpg.mvThemeCol_Text,          (255, 255, 255))
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10)
        dpg.add_theme_style(dpg.mvStyleVar_FramePadding,  14,  7)



with dpg.texture_registry(show=False, tag="global_texture_registry"):
    pass

with dpg.font_registry():
    with dpg.font("Roboto-Regular.ttf", 15) as default_font:
        pass
    dpg.bind_font(default_font)

with dpg.file_dialog(directory_selector=False, show=False, callback=file_selected_cb, tag="file_dialog_id", width=800, height=600):
    dpg.add_file_extension(".json", color=(0, 120, 215, 255))
    dpg.add_file_extension(".*", color=(50, 50, 50, 255))

# Training popup – light styled
with dpg.window(label="Training Parameters", modal=True, show=False, tag="training_popup",
                no_title_bar=False, width=420, height=360):
    dpg.add_text("Training Configuration", color=[92, 75, 192])
    dpg.add_separator()
    dpg.add_spacer(height=6)

    dpg.add_input_int(  label="Iterations",       tag="input_iters",         width=210)
    dpg.add_input_float(label="Learning Rate",     tag="input_lr",            width=210, format="%.4f", step=0.001)
    dpg.add_input_float(label="Bound",             tag="input_bound",         width=210, format="%.1f")
    dpg.add_input_float(label="Scale",             tag="input_scale",         width=210, format="%.2f")
    dpg.add_input_float(label="dt_gamma",          tag="input_dt_gamma",      width=210, format="%.4f", step=0.001)
    dpg.add_input_int(  label="Max Steps",         tag="input_max_steps",     width=210)
    dpg.add_input_float(label="Density Threshold", tag="input_density_thresh", width=210, format="%.1f")
    dpg.add_checkbox(   label="Show GUI during training", tag="input_gui")

    dpg.add_spacer(height=14)
    with dpg.group(horizontal=True):
        dpg.add_button(label="Start Training", width=140, height=32, callback=confirm_training_cb)
        dpg.add_button(label="Cancel",             width=90,  height=32, callback=cancel_training_cb)
with dpg.theme() as global_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,  9)
        dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 12)
        dpg.add_theme_style(dpg.mvStyleVar_ChildRounding,  9)
        dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 10)
        dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,  20, 16)
        dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,    10,  8)

        # Backgrounds
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg,  (246, 247, 252))
        dpg.add_theme_color(dpg.mvThemeCol_PopupBg,   (255, 255, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg,   (250, 251, 255))

        # Text
        dpg.add_theme_color(dpg.mvThemeCol_Text,          ( 28,  30,  50))
        dpg.add_theme_color(dpg.mvThemeCol_TextDisabled,   (160, 165, 185))

        # Borders
        dpg.add_theme_color(dpg.mvThemeCol_Border,        (215, 218, 232))
        dpg.add_theme_color(dpg.mvThemeCol_BorderShadow,  (  0,   0,   0,   0))

        # Inputs
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg,        (255, 255, 255))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered,  (242, 244, 255))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive,   (232, 236, 252))

        # Buttons (default)
        dpg.add_theme_color(dpg.mvThemeCol_Button,        (236, 238, 248))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,  (222, 226, 244))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,   (210, 215, 238))

        # Scrollbar
        dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg,          (240, 241, 248))
        dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab,         (200, 205, 225))
        dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered,  (175, 182, 210))
        dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive,   (148, 156, 195))

        # Title bar
        dpg.add_theme_color(dpg.mvThemeCol_TitleBg,         (235, 237, 248))
        dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,   (222, 226, 244))
        dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed,(240, 241, 248))

        # Checkbox, slider
        dpg.add_theme_color(dpg.mvThemeCol_CheckMark,       ( 92,  75, 192))
        dpg.add_theme_color(dpg.mvThemeCol_SliderGrab,      (112,  95, 215))
        dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive,( 74,  60, 165))
        dpg.add_theme_color(dpg.mvThemeCol_Header,           (228, 232, 248))
        dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,    (215, 220, 242))
        dpg.add_theme_color(dpg.mvThemeCol_HeaderActive,     (200, 208, 235))



with dpg.window(label="NeRF Model Manager", tag="primary_window", width=870, height=790):
    # ─── Header ──────────────────────────────────────────────────────────────
    with dpg.group(horizontal=True):
        dpg.add_text("3D Model Manager", color=[ 30,  35,  70])
        dpg.add_text("  -  Instant-NGP", color=[155, 160, 185])
        btn_add_new = dpg.add_button(
            label="+  Add Dataset",
            callback=lambda: dpg.show_item("file_dialog_id"),
            width=148, height=30, indent=572
        )
    dpg.add_separator()
    dpg.add_spacer(height=4)

    # ─── Grid ────────────────────────────────────────────────────────────────
    with dpg.child_window(width=-1, height=648, tag="history_grid", border=False):
        pass

    dpg.add_separator()
    dpg.add_spacer(height=2)

    # ─── Status bar ──────────────────────────────────────────────────────────
    with dpg.group(horizontal=True):
        dpg.add_text("Status: Ready", tag="status_text", color=[120, 128, 165])
        dpg.add_spacer(width=40)
        dpg.add_text("Server: Stopped", tag="server_status_text", color=[185,  80,  90])

dpg.bind_item_theme(btn_add_new, "add_btn_theme")

dpg.bind_theme(global_theme)

render_history_grid()

dpg.create_viewport(title='NeRF Model Launcher', width=870, height=790)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("primary_window", True)
dpg.set_exit_callback(cleanup_all_processes)
dpg.start_dearpygui()
dpg.destroy_context()

