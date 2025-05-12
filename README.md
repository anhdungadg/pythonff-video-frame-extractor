# Video Frame Extractor

Công cụ trích xuất frame từ video theo tỷ lệ phần trăm được chỉ định.

## Mô tả

Chương trình Python này cho phép người dùng trích xuất một tỷ lệ phần trăm nhất định các frame từ file video. Thay vì trích xuất tất cả các frame, công cụ này cho phép bạn chỉ định tỷ lệ phần trăm số frame cần lấy, giúp tiết kiệm không gian lưu trữ và thời gian xử lý.

## Tính năng

- Trích xuất frame từ video theo tỷ lệ phần trăm được chỉ định
- Tự động tạo thư mục đầu ra nếu chưa tồn tại
- Hiển thị thông tin về video (tổng số frame, FPS)
- Hiển thị tiến trình trích xuất
- Lưu các frame dưới dạng file JPG với tên được đánh số

## Yêu cầu

- Python 3.x
- OpenCV (cv2)

Cài đặt thư viện cần thiết:

```bash
pip install opencv-python
```

## Cách sử dụng

```bash
python extract_frames.py đường_dẫn_video thư_mục_lưu_frame tỷ_lệ_phần_trăm
```

### Tham số

- `đường_dẫn_video`: Đường dẫn đến file video cần trích xuất frame
- `thư_mục_lưu_frame`: Thư mục để lưu các frame được trích xuất
- `tỷ_lệ_phần_trăm`: Tỷ lệ phần trăm số frame cần trích xuất (từ 0 đến 100)

### Ví dụ

```bash
python extract_frames.py video.mp4 output_frames 10
```

Lệnh trên sẽ trích xuất 10% số frame từ file video.mp4 và lưu vào thư mục output_frames.

## Cách hoạt động

1. Đọc file video và xác định tổng số frame
2. Tính toán số frame cần trích xuất dựa trên tỷ lệ phần trăm
3. Tính khoảng cách giữa các frame cần trích xuất để đảm bảo phân bố đều
4. Trích xuất và lưu các frame theo khoảng cách đã tính
5. Hiển thị tiến trình và thông báo khi hoàn thành

## Định dạng đầu ra

Các frame được lưu dưới dạng file JPG với tên theo định dạng `frame_XXXX.jpg`, trong đó XXXX là số thứ tự của frame được trích xuất (bắt đầu từ 0000).

## Ứng dụng

- Tạo bộ dữ liệu hình ảnh từ video
- Phân tích nội dung video
- Tạo thumbnails hoặc preview cho video
- Xử lý và phân tích hình ảnh từ video
