import bpy
import math
import mathutils
import json
import os
import random

# ==========================================
# CẤU HÌNH THÔNG SỐ DATASET
# ==========================================
# Thư mục lưu dataset (thay đổi theo máy của bạn)
OUTPUT_DIR = "/tmp/nerf_dataset"  

NUM_IMAGES = 50           # Số lượng ảnh muốn render
RADIUS_MIN = 3.5          # Bán kính nhỏ nhất (khoảng cách từ camera đến vật thể)
RADIUS_MAX = 4.5          # Bán kính lớn nhất
RESOLUTION = 800          # Độ phân giải ảnh (vuông)
HEMISPHERE_ONLY = True    # True: Chỉ xoay camera ở nửa trên (Z > 0). False: Cả khối cầu.

# ==========================================

def setup_scene():
    """Thiết lập thông số cho Scene"""
    scene = bpy.context.scene
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA' # Lưu nền trong suốt nếu có

def get_random_camera_loc():
    """Tạo vị trí camera ngẫu nhiên xoay quanh tâm vật thể"""
    theta = random.uniform(0, 2 * math.pi)
    
    if HEMISPHERE_ONLY:
        # Nửa trên của mặt cầu
        phi = math.acos(random.uniform(0.0, 1.0))
    else:
        # Toàn bộ mặt cầu
        phi = math.acos(random.uniform(-1.0, 1.0))
        
    r = random.uniform(RADIUS_MIN, RADIUS_MAX)
    
    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    
    return mathutils.Vector((x, y, z))

def look_at(obj_camera, target_loc):
    """Xoay camera hướng thẳng về phía một điểm (ví dụ tâm 0,0,0)"""
    direction = target_loc - obj_camera.location
    # Blender camera nhìn theo trục -Z, hướng lên là trục Y
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj_camera.rotation_euler = rot_quat.to_euler()

def generate_dataset():
    # Tạo thư mục
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images_dir = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    setup_scene()
    
    # Lấy camera hiện tại, nếu chưa có thì tạo
    cam_obj = bpy.context.scene.camera
    if cam_obj is None:
        cam_data = bpy.data.cameras.new("NeRFCam")
        cam_obj = bpy.data.objects.new("NeRFCam", cam_data)
        bpy.context.collection.objects.link(cam_obj)
        bpy.context.scene.camera = cam_obj
        
    cam = cam_obj.data
    camera_angle_x = cam.angle_x
    
    frames = []
    
    print(f"Bắt đầu render {NUM_IMAGES} ảnh...")
    
    for i in range(NUM_IMAGES):
        # 1. Đặt vị trí ngẫu nhiên
        cam_loc = get_random_camera_loc()
        cam_obj.location = cam_loc
        
        # 2. Hướng vào tâm gốc tọa độ (nơi đặt vật thể)
        look_at(cam_obj, mathutils.Vector((0.0, 0.0, 0.0)))
        
        # Cập nhật scene để tính toán ma trận mới
        bpy.context.view_layer.update()
        
        # 3. Render ảnh
        img_name = f"r_{i:03d}"
        img_path = os.path.join(images_dir, f"{img_name}.png")
        bpy.context.scene.render.filepath = img_path
        bpy.ops.render.render(write_still=True)
        
        # 4. Lấy ma trận Camera to World của Blender 
        # (Instant-NGP / NeRF tự động nhận diện chuẩn OpenGL mà ma trận Blender khá tương thích)
        matrix = cam_obj.matrix_world
        list_matrix = [list(row) for row in matrix]
        
        # 5. Lưu vào frames
        frames.append({
            "file_path": f"./images/{img_name}.png",
            "transform_matrix": list_matrix
        })
        
        print(f"[{i+1}/{NUM_IMAGES}] Đã render {img_name}.png")
        
    # Lưu file transforms.json
    transforms = {
        "camera_angle_x": camera_angle_x,
        "frames": frames
    }
    
    transforms_path = os.path.join(OUTPUT_DIR, "transforms.json")
    with open(transforms_path, 'w', encoding='utf-8') as f:
        json.dump(transforms, f, indent=4)
        
    print(f"\n HOÀN TẤT! Dataset đã được lưu tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_dataset()
