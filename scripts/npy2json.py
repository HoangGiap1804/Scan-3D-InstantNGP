import os
import sys
import json
import numpy as np
import argparse

def convert_llff_to_json(dataset_dir):
    npy_path = os.path.join(dataset_dir, 'poses_bounds.npy')
    if not os.path.exists(npy_path):
        print(f"Error: {npy_path} not found.")
        sys.exit(1)
        
    print(f"Loading {npy_path}...")
    poses_arr = np.load(npy_path)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])
    bounds = poses_arr[:, -2:]
    
    H, W, focal = poses[0, :, -1]
    
    # Tìm danh sách ảnh
    img_dir = os.path.join(dataset_dir, 'images')
    if os.path.exists(img_dir):
        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    else:
        print(f"Warning: images folder not found. Using dummy names.")
        img_files = [f"img_{i:04d}.png" for i in range(len(poses))]
        
    if len(img_files) != len(poses):
        print(f"Warning: Number of images ({len(img_files)}) does not match number of poses ({len(poses)}).")
        
    frames = []
    
    for i in range(len(poses)):
        pose = poses[i]
        
        c2w = np.eye(4)
        # LLFF poses: [down, right, backwards]
        # NeRF poses: [right, up, backwards]
        c2w[:3, 0] = pose[:, 1]   # right
        c2w[:3, 1] = -pose[:, 0]  # up = -down
        c2w[:3, 2] = pose[:, 2]   # backwards
        c2w[:3, 3] = pose[:, 3]   # translation
        
        file_path = f"images/{img_files[i]}" if i < len(img_files) else f"images/img_{i:04d}.png"
        
        frames.append({
            "file_path": file_path,
            "transform_matrix": c2w.tolist()
        })
        
    out = {
        "fl_x": float(focal),
        "fl_y": float(focal),
        "cx": float(W / 2.0),
        "cy": float(H / 2.0),
        "w": int(W),
        "h": int(H),
        "frames": frames
    }
    
    out_path = os.path.join(dataset_dir, 'transforms.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=4)
        
    print(f"Successfully generated {out_path} with {len(frames)} frames!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help="Path to the dataset directory containing poses_bounds.npy")
    args = parser.parse_args()
    
    convert_llff_to_json(args.path)
