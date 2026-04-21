# Giải thích chi tiết thuật toán trong hàm `update_extra_state`

Hàm `update_extra_state()` nằm trong `renderer.py` chính là "bộ não điều phối tốc độ" của mô hình **Instant-NGP**. 
Nguyên lý tính toán NeRF thông thường rất chậm vì các tia sáng (rays) phải lấy mẫu mù mờ dọc theo các điểm trống rỗng. Hàm này sinh ra để vẽ ra một bản đồ khoanh vùng "Occupancy Grid" giúp tia sáng biết chỗ nào có lõi mô hình thì mới bắt đầu tính toán màu sắc, chỗ nào trống rỗng thì mạnh dạn nhảy qua (Empty-space skipping).

Dưới đây là phần giải thích chi tiết hoạt động của từng dòng code:

---

### 1. Điều kiện tiền đề & Khởi tạo (Line 411 - 417)
```python
if not self.cuda_ray: return 
tmp_grid = - torch.ones_like(self.density_grid)
```
- **Line 411-412:** Thuật toán này chỉ chạy nếu đang bật chế độ `cuda_ray=True` (sử dụng CUDA cho ray marching phân mảnh). Nếu không, nó sẽ bị bỏ qua.
- **Line 416:** Khởi tạo một mảng lưới tạm (`tmp_grid`) ngập số -1 để làm bộ đệm tính trước giá trị không không gian rỗng (tương đương với việc phủ nhận mật độ).

### 2. Giai đoạn Full Update (Line 419 - 447)
**Trường hợp xảy ra:** Trong 16 lần cập nhật đầu tiên (`iter_density < 16`), hệ thống không biết cấu trúc 3D có hình thù gì do ban đầu weight mô hình là ngẫu nhiên, nên phải quét tổng thể toàn bộ không gian.
- **Line 420-432:** Nó băm nhỏ không gian ra bằng mạng lưới tọa độ tọa bởi `custom_meshgrid`.
- **Line 431:** Sử dụng `raymarching.morton3D`. (Đường cong Z-order Morton) Đây là kỹ thuật biến đổi không gian tọa độ XYZ (3 chiều) xuống chuỗi (1 chiều) nhằm tăng tốc độ truy cập bộ nhớ cache lúc sử dụng CUDA.
- **Line 435-447:** Có một vòng lặp `for cas in range(self.cascade):`. Mạng lưới phân tầng (Cascade) như quả cầu bọc nhau: lõi trung tâm độ phân giải cực cao, càng ra viền ngoài độ phân giải càng thưa hơn. Đoạn này tính toán tỷ lệ, **chủ động chèn nhiễu** (add noise `+ rand() * half_grid_size`) để tránh răng cưa.
- **Line 443:** Gọi `self.density(cas_xyzs)` để ép Mạng Nơ ron tính mật độ xuyên qua lưới tĩnh. Nó sẽ lưu mật độ tạm vào lưới `tmp_grid`.

### 3. Giai đoạn Partial Update (Line 449 - 476)
**Trường hợp xảy ra:** Nếu mạng đã chạy qua pha khỏi động ban đầu (iter > 16), bây giờ không tội gì mỗi step lại đi quét Full Grid nữa vì tốn VRAM. Nó tối ưu bằng cách truy vấn một bộ phận nhỏ mẫu:
- **Line 453-454:** Tính random ngẫu nhiên $1/4$ số điểm trong lưới.
- **Line 456-460:** Lấy thêm $1/4$ số điểm khác nhưng **chỉ được phép rơi vào các vùng đang dầy đặc có mật độ trước đó** (`occ_indices = torch.nonzero(density_grid > 0)`). Cách kết hợp này vừa dò được viền của vật thể, vừa giữ độ chi tiết cao tại vị trí bề mặt.
- **Line 461-476:** Gộp hai tập mẫu này lại và nhét vào mạng báo mật độ `self.density(...)` như tương tự pha số (2).

### 4. Giai đoạn áp dụng EMA (Exponential Moving Average) (Line 478 - 482)
```python
self.density_grid[valid_mask] = torch.maximum(
    self.density_grid[valid_mask] * decay, 
    tmp_grid[valid_mask]
)
```
- Kết quả thu được nếu đem gán chép đè thẳng đi sẽ làm lưới chớp nháy (bất ổn khi render do bị noise). 
- Do đó, để lưới mượt, nó lấy giá trị lớn nhất cực đại tạm thời (`torch.maximum`). Song song với đó, các vùng được update cũ sẽ tự động tan mờ đần vì hệ số phân rã `decay = 0.95`. Nghĩa là nếu sau 1 đống steps không thấy có khối đặc ở đó nữa, mật độ lưới mờ sẽ tự rớt xuống giá trị rỗng chứ không đột ngột thủng lỗ.

### 5. Gói cước C++ Cache bằng `Bitfield` (Line 484 - 486)
```python
density_thresh = min(self.mean_density, self.density_thresh)
self.density_bitfield = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)
```
- Khởi điểm lõi `cuda_raymarching` bằng C++ sử dụng kỹ thuật truyền bit nhị phân để bỏ qua tia. Cho nên mảng float32 Tensor khổng lồ sẽ được đem so với một `density_thresh`.
- Cứ hễ mật độ lưới > ngưỡng này thì tức là bằng 1, dưới thì bằng 0. 
- Ngay sau đó `packbits` nén chặt 8 tín hiệu float 0-1 vào làm một thẻ `[uint8]` duy nhất. Việc nén này tiết kiệm năng lực tính toán cực đoan và đẩy độ lớn batch-rays mà GPU nhồi được trong một frame lên đến ngưỡng chục nghìn.

### 6. Thống kê theo dõi Step (Line 488 - 492)
```python
self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
self.local_step = 0
```
- Reset và cập nhật thông số `mean_count`, giúp theo dõi báo cáo xem mỗi tia trung bình đã nhảy qua bao nhiêu steps.

## Tổng Kết

Về cơ bản đoạn block `update_extra_state` này xử lý hoàn hảo việc:
1. Tạo một mạng lưới thô băm không gian để dò tìm bề mặt (Mô phỏng tia chụp MRI ảo).
2. Tối ưu cực nhanh (Chỉ tính Full lúc 16 iterations đầu, sau đó chỉ update mẫu và nơi có vật thể).
3. Đóng băng thành 2 bit nhị phân chuyển đổi thành thẻ C++ gửi cho CUDA Kernel thực hiện Empty-space skipping tốc độ kinh hoàng.
