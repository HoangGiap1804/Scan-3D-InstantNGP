# Scan3D-InstantNGP

Dự án triển khai mô hình Neural Radiance Fields (NeRF) hiệu năng cao dựa trên kiến trúc Instant-NGP. Dự án sử dụng bộ mã hóa Hash Grid và CUDA ray marching để đạt được tốc độ huấn luyện và render cực nhanh.

## 🚀 Tính năng chính

*   **Tốc độ cao**: Sử dụng thư viện `tiny-cuda-nn` và nhân CUDA tùy chỉnh.
*   **Trình xem tương tác (GUI)**: Tích hợp giao diện DearPyGui để xem kết quả 3D trong thời gian thực.
*   **Ổn định**: Triển khai gradient clipping và scheduler giảm tốc độ học liên tục để tránh lỗi `NaN`.
*   **Linh hoạt**: Hỗ trợ tập dữ liệu dạng chuẩn (COLMAP/Transforms.json).

## 🛠️ Yêu cầu hệ thống

*   **GPU**: NVIDIA GPU với kiến trúc Pascal trở lên (Khuyến nghị RTX 20 series trở lên).
*   **Hệ điều hành**: Linux (Khuyến nghị Ubuntu hoặc NixOS).
*   **Phần mềm**:
    *   Nix (tùy chọn nhưng khuyến nghị)
    *   CUDA Toolkit 11+
    *   Python 3.10+

## 📦 Cài đặt

### Cách 1: Sử dụng Nix (Khuyến nghị)

Nếu bạn cài đặt Nix, chỉ cần chạy lệnh sau để thiết lập môi trường:

```bash
nix-shell
```

Sau khi vào shell, hãy cài đặt các thư viện Python:

```bash
pip install -r requirements.txt
```

### Cách 2: Sử dụng pip thông thường

Đảm bảo bạn đã cài đặt CUDA và NVCC. Chạy:

```bash
pip install -r requirements.txt
```

*Lưu ý: Bạn có thể cần cài đặt thêm `tiny-cuda-nn` thủ công nếu gặp lỗi build.*

## 🏃 Cách chạy chương trình

### 1. Huấn luyện mô hình (Training)

Để bắt đầu huấn luyện với tập dữ liệu của bạn, sử dụng lệnh sau:

```bash
python train.py --path ./data/fox --workspace workspace_fox --epochs 200 --bound 0.5 --fp16 
```

Các tham số chính:
*   `--path`: Đường dẫn đến thư mục chứa dữ liệu (có file `transforms.json`).
*   `--workspace`: Thư mục lưu checkpoint và kết quả validation.
*   `--bound`: Giới hạn cảnh (mặc định là 2.0).
*   `--lr`: Tốc độ học (mặc định 5e-3).

### 2. Xem kết quả tương tác (GUI Viewer)

Sau khi huấn luyện (hoặc muốn xem checkpoint có sẵn), hãy chạy:

```bash
python gui.py --workspace workspace_fox --res 400 --display 800
```

Các tham số GUI:
*   `--res`: Độ phân giải render (thấp để tăng FPS).
*   `--display`: Độ phân giải hiển thị (tự động upscale từ `res`).
*   `--ckpt`: Đường dẫn cụ thể đến file `.pth` (nếu không sẽ lấy file mới nhất trong workspace).

## 📂 Cấu trúc thư mục

*   `train.py`: Script huấn luyện chính.
*   `gui.py`: Công cụ xem 3D tương tác.
*   `model.py`: Định nghĩa kiến trúc mạng NeRF.
*   `renderer.py`: Logic render tia (ray marching).
*   `provider.py`: Xử lý tập dữ liệu.
*   `utils.py`: Các hàm bổ trợ (render ảnh, lưu video, v.v.).
*   `shell.nix`: Cấu hình môi trường cho NixOS.

## ⚠️ Lưu ý

*   **Lỗi OOM**: Nếu gặp lỗi tràn bộ nhớ GPU, hãy giảm `--num_rays` trong `train.py` hoặc giảm độ phân giải `--res` trong `gui.py`.
*   **NaN Loss**: Đã được khắc phục bằng gradient clipping. Nếu vẫn gặp phải, hãy thử giảm `--lr`.
