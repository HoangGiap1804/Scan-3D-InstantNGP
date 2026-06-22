import os
import glob
import json
import numpy as np
import math
import argparse

def parse_intrinsics(intrinsics_path):
    with open(intrinsics_path, 'r') as f:
        lines = f.readlines()
    
    # First line usually contains: focal cx cy ...
    vals = lines[0].strip().split()
    fl_x = float(vals[0])
    fl_y = fl_x  # Assuming square pixels
    
    # Check if the last line contains W and H
    w, h = 800, 800
    for line in reversed(lines):
        vals = line.strip().split()
        if len(vals) == 2:
            w = int(vals[0])
            h = int(vals[1])
            break
            
    # Assuming cx, cy are in the first line if it has enough elements, 
    # otherwise default to w/2, h/2
    vals = lines[0].strip().split()
    if len(vals) >= 3:
        cx = float(vals[1])
        cy = float(vals[2])
    else:
        cx = w / 2.0
        cy = h / 2.0
        
    return fl_x, fl_y, cx, cy, w, h

def convert_nsvf_to_nerf(dataset_dir, output_dir=None):
    if output_dir is None:
        output_dir = dataset_dir

    intrinsics_path = os.path.join(dataset_dir, 'intrinsics.txt')
    if not os.path.exists(intrinsics_path):
        print(f"Error: {intrinsics_path} not found.")
        return
        
    fl_x, fl_y, cx, cy, w, h = parse_intrinsics(intrinsics_path)
    camera_angle_x = math.atan(w / (fl_x * 2)) * 2
    camera_angle_y = math.atan(h / (fl_y * 2)) * 2
    
    print(f"Parsed intrinsics: fl_x={fl_x}, fl_y={fl_y}, cx={cx}, cy={cy}, w={w}, h={h}")
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        frames = []
        pose_files = glob.glob(os.path.join(dataset_dir, 'pose', f'*_{split}_*.txt'))
        pose_files.sort()
        
        for pose_file in pose_files:
            basename = os.path.basename(pose_file)
            prefix = basename.replace('.txt', '')
            
            # Read pose
            pose = np.loadtxt(pose_file)
            
            # NSVF poses are OpenCV C2W (X right, Y down, Z forward).
            # NeRF (Instant-NGP) uses OpenGL C2W (X right, Y up, Z backward).
            # We convert by flipping the Y and Z axes.
            pose[0:3, 1:3] *= -1
            
            img_path = f"./rgb/{prefix}.png"
            
            frames.append({
                "file_path": img_path,
                "transform_matrix": pose.tolist()
            })
            
        if not frames:
            print(f"No frames found for split {split}.")
            continue
            
        out_dict = {
            "camera_angle_x": camera_angle_x,
            "camera_angle_y": camera_angle_y,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "frames": frames
        }
        
        out_path = os.path.join(output_dir, f"transforms_{split}.json")
        with open(out_path, 'w') as f:
            json.dump(out_dict, f, indent=4)
            
        print(f"Saved {len(frames)} frames to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert NSVF dataset to NeRF transforms.json format")
    parser.add_argument('dataset_dir', type=str, help="Path to the NSVF dataset directory")
    parser.add_argument('--output_dir', type=str, default=None, help="Output directory for transforms JSONs (default: same as dataset_dir)")
    args = parser.parse_args()
    
    convert_nsvf_to_nerf(args.dataset_dir, args.output_dir)
