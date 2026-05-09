# Scan-3D-InstantNGP

Dự án tái tạo vật thể 3D từ Video/Hình ảnh sử dụng Instant-NGP.

## 1. Cài đặt môi trường
Đảm bảo bạn đã cài đặt Python 3.9+ và các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## 2. Khởi chạy Giao diện (GUI) & Server
Giao diện này tích hợp cả trình xem 3D và Server nhận dữ liệu từ điện thoại.
```bash
python gui.py --res 400 --workspace ./workspace
```
*   **Cổng 8000**: WebSocket Viewer (Xem 3D thời gian thực).
*   **Cổng 8080**: HTTP API (Nhận video upload).
*   *Lưu ý: Quét mã QR trên Console để lấy địa chỉ IP.*

## 3. Xử lý dữ liệu (COLMAP)
Chuyển đổi Video/Hình ảnh sang định dạng NeRF.

### Từ Video (Khuyên dùng):
```bash
python scripts/colmap2nerf.py --video ./videos/test.mp4 --run_colmap --video_fps 2 --size 800x800
```
*   `--video`: Đường dẫn file video.
*   `--run_colmap`: Chạy tiến trình tính toán vị trí camera.
*   `--video_fps 2`: Tốc độ lấy mẫu ảnh (2 ảnh mỗi giây).
*   `--size 800x800`: Giảm kích thước ảnh để tăng tốc độ xử lý và training.

### Từ thư mục ảnh:
```bash
python scripts/colmap2nerf.py --images ./data/my_folder --run_colmap
```

## 4. Huấn luyện mô hình (Training)
Nếu bạn không dùng tính năng tự động xử lý trên GUI, bạn có thể chạy bằng tay:
```bash
python train.py --path ./data/custom_data --workspace ./workspace --epochs 100 --fp16
```

## 5. API dành cho Mobile App
### Upload Video
*   **URL**: `http://<SERVER_IP>:8080/upload`
*   **Method**: `POST`
*   **Body**: `multipart/form-data`
*   **Field**: `file` (chọn file video)

### Viewer (WebSocket)
*   **URL**: `ws://<SERVER_IP>:8000`
*   **Data**: JSON chứa `type: "image"` để nhận ảnh render.

## Các lưu ý quan trọng:
*   **CUDA**: COLMAP sẽ tự động dùng CUDA nếu có. Đừng dùng tham số `--estimate_affine_shape` nếu muốn chạy nhanh nhất trên GPU.
*   **Thư mục Videos**: Các file upload sẽ được lưu vào thư mục `videos/` với cấu trúc thư mục riêng biệt cho từng lần upload.
