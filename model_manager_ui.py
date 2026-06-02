import dearpygui.dearpygui as dpg
import subprocess
import shlex
import threading
import json
import os

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

def file_selected_cb(sender, app_data):
    file_path = app_data.get('file_path_name', "")
    if file_path:
        if file_path in history_list:
            history_list.remove(file_path)
        history_list.insert(0, file_path) 
        
        if len(history_list) > 12:
            history_list.pop()
            
        save_history(history_list)
        render_history_grid()
        run_model_callback(file_path)

def history_button_cb(sender, app_data, user_data):
    file_path = user_data
    if file_path in history_list:
        history_list.remove(file_path)
    history_list.insert(0, file_path)
    save_history(history_list)
    render_history_grid()
    run_model_callback(file_path)

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
    dpg.delete_item("history_grid", children_only=True)
    
    if not history_list:
        dpg.add_text("No recent files yet.", parent="history_grid", color=[150, 150, 150])
        return

    table_id = dpg.add_table(header_row=False, parent="history_grid", borders_innerH=False, borders_innerV=False, borders_outerH=False, borders_outerV=False)
    dpg.add_table_column(parent=table_id)
    dpg.add_table_column(parent=table_id)
    dpg.add_table_column(parent=table_id)
    dpg.add_table_column(parent=table_id)
    
    row_id = None
    for i, file_path in enumerate(history_list):
        if i % 4 == 0:
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
        with dpg.child_window(parent=row_id, width=175, height=215, border=True, no_scrollbar=True) as card:
            # Card theme will apply 12px padding, leaving exactly 150px for content (175 - 24 = 151)
            if tex_id:
                img_btn = dpg.add_image_button(tex_id, width=150, height=150, callback=history_button_cb, user_data=file_path)
            else:
                img_btn = dpg.add_button(label="[ No Image ]", width=150, height=150, callback=history_button_cb, user_data=file_path)
            
            # Label dưới hình
            txt_btn = dpg.add_button(label=display_name, width=150, height=35, callback=history_button_cb, user_data=file_path)
            
            dpg.bind_item_theme(card, "card_theme")
            dpg.bind_item_theme(img_btn, "image_btn_theme")
            dpg.bind_item_theme(txt_btn, "text_btn_theme")

def run_model_callback(file_path):
    if not file_path:
        return
        
    dir_path = os.path.dirname(file_path).replace('\\', '/')
    new_workspace = os.path.basename(dir_path) or "my_model"
    cmd = f'python main_nerf.py "{dir_path}" --workspace workspaces/{new_workspace} -O --bound 1.0 --scale 0.7 --dt_gamma 0 --color_space linear --error_map --gui'
        
    dpg.set_value("status_text", f"Status: Launching model {new_workspace}...")
    
    def execute():
        try:
            args = shlex.split(cmd)
            subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            dpg.set_value("status_text", "Status: Command executed successfully! 3D window will appear.")
        except Exception as e:
            dpg.set_value("status_text", f"Execution error: {str(e)}")
            
    threading.Thread(target=execute, daemon=True).start()


dpg.create_context()

with dpg.theme(tag="card_theme"):
    with dpg.theme_component(dpg.mvChildWindow):
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (255, 255, 255))
        dpg.add_theme_color(dpg.mvThemeCol_Border, (220, 225, 230))
        dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 10)
        dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 12)
        dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 4, 4)

with dpg.theme(tag="image_btn_theme"):
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0)) # Trong suốt để ẩn nền xám xấu xí
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (245, 245, 245))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (235, 235, 235))
        dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0)
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)

with dpg.theme(tag="text_btn_theme"):
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (244, 248, 252))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (230, 240, 250))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (210, 230, 245))
        dpg.add_theme_color(dpg.mvThemeCol_Text, (20, 80, 160))
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)

with dpg.texture_registry(show=False, tag="global_texture_registry"):
    pass

with dpg.font_registry():
    with dpg.font("Roboto-Regular.ttf", 18) as default_font:
        pass
    dpg.bind_font(default_font)

with dpg.file_dialog(directory_selector=False, show=False, callback=file_selected_cb, tag="file_dialog_id", width=750, height=500):
    dpg.add_file_extension(".json", color=(0, 120, 215, 255))
    dpg.add_file_extension(".*", color=(50, 50, 50, 255))

# --- Global Light Theme ---
with dpg.theme() as global_theme:
    with dpg.theme_component(dpg.mvAll):
        # Rounding cho các thành phần tạo cảm giác hiện đại (modern UI)
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
        dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 8)
        dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 8)
        dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 20, 20)
        dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 12, 12)
        
        # Cửa sổ chính màu nền xám rất nhạt (như Windows/Mac)
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (244, 246, 248))
        dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (255, 255, 255)) 
        
        # Nền child (Card) là màu trắng tinh để làm nổi bật card
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (255, 255, 255)) 
        dpg.add_theme_color(dpg.mvThemeCol_Text, (30, 30, 30))
        dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (130, 130, 130))
        dpg.add_theme_color(dpg.mvThemeCol_Border, (220, 225, 230))
        
        # Ô nhập lệnh (FrameBg)
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (255, 255, 255))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (245, 248, 255))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (235, 245, 255))
        
        # Các nút bấm thông thường (màu xanh nhạt)
        dpg.add_theme_color(dpg.mvThemeCol_Button, (240, 245, 250))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (225, 238, 250))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (200, 225, 245))
        
        dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, (240, 240, 240))
        dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, (200, 200, 200))
        dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, (170, 170, 170))
        dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, (140, 140, 140))

with dpg.window(label="NeRF Model Manager", tag="primary_window", width=850, height=780):
    dpg.add_text("3D Model Manager (Instant-NGP)", color=[0, 80, 180])
    dpg.add_spacer(height=5)
    
    dpg.add_text("Recent selected files (Click image card to RUN immediately):", color=[0, 120, 50])
    # Tăng chiều cao lên 250 để chứa được 1 hàng rưỡi (hiện thanh cuộn nhẹ)
    with dpg.child_window(width=-1, height=260, tag="history_grid", border=False):
        pass
    
    dpg.add_spacer(height=10)
    dpg.add_button(label="BROWSE AND SELECT NEW transforms.json FILE", callback=lambda: dpg.show_item("file_dialog_id"), width=-1, height=45)
    
    dpg.add_spacer(height=20)
    dpg.add_text("Status: Ready", tag="status_text", color=[130, 130, 130])

dpg.bind_theme(global_theme)

render_history_grid()

dpg.create_viewport(title='NeRF Model Launcher', width=850, height=750)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("primary_window", True)
dpg.start_dearpygui()
dpg.destroy_context()
