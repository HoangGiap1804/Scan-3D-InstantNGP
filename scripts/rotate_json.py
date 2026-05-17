import json
import numpy as np
import argparse
import os

def rotate_json(json_path, output_path=None):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Ma trận xoay: Biến trục Y cũ thành -Z mới (Xoay -90 độ quanh trục X)
    # [1, 0,  0]
    # [0, 0,  1]
    # [0, -1, 0]
    rot = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    for frame in data['frames']:
        c2w = np.array(frame['transform_matrix'])
        # Thực hiện xoay hệ tọa độ cho từng pose
        new_c2w = rot @ c2w
        frame['transform_matrix'] = new_c2w.tolist()

    if output_path is None:
        output_path = json_path.replace('.json', '_rotated.json')

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Successfully rotated coordinate system (Y -> -Z)!")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help="Path to transforms.json")
    args = parser.parse_args()
    
    rotate_json(args.path)
