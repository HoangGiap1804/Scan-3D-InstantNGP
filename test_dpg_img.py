import dearpygui.dearpygui as dpg
import os
import json

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
    except:
        pass
    return None

dpg.create_context()
img_path = get_first_image_for_dataset("data/nerf_synthetic/chair/transforms_train.json")
print("Found image:", img_path)
if img_path:
    w, h, c, data = dpg.load_image(img_path)
    print(f"Loaded: {w}x{h}, channels: {c}, data size: {len(data)}")
    if len(data) == w*h*4:
        print("Data is 4 channels (RGBA). Safe for DPG!")
    else:
        print("Data is NOT 4 channels. Need manual conversion.")
dpg.destroy_context()
