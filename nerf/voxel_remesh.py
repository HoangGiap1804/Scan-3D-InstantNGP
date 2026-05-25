import bpy
import sys

# Lấy các tham số truyền vào từ terminal sau dấu '--'
argv = sys.argv
try:
    argv = argv[argv.index("--") + 1:] 
except ValueError:
    print("Lỗi: Thiếu ký tự '--' phân cách tham số.")
    sys.exit(1)

input_file = argv[0]
output_file = argv[1]

# 1. Dọn dẹp toàn bộ scene mặc định
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# 2. Import file .ply (Sử dụng API mới của Blender 4.x)
print(f"Đang import: {input_file}...")
bpy.ops.wm.ply_import(filepath=input_file)

# Chọn object vừa được import
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

# 3. Thêm và Apply Modifier Voxel Remesh
print("Đang xử lý Voxel Remesh...")
remesh_mod = obj.modifiers.new(name="VoxelRemesh", type='REMESH')
remesh_mod.mode = 'VOXEL'
remesh_mod.voxel_size = 0.01
remesh_mod.adaptivity = 0.0

bpy.ops.object.modifier_apply(modifier=remesh_mod.name)

# 4. Export ra file .ply mới (Sử dụng API mới của Blender 4.x)
print(f"Đang export: {output_file}...")
bpy.ops.wm.ply_export(filepath=output_file)

print(f"Hoàn tất! Đã lưu file: {output_file}")