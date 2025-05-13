# Video Frame Extractor

Công cụ trích xuất frame từ video theo tỷ lệ phần trăm được chỉ định và phát hiện cảnh cắt trong video.

## Mô tả

Chương trình Python này cho phép người dùng trích xuất một tỷ lệ phần trăm nhất định các frame từ file video. Thay vì trích xuất tất cả các frame, công cụ này cho phép bạn chỉ định tỷ lệ phần trăm số frame cần lấy, giúp tiết kiệm không gian lưu trữ và thời gian xử lý. Ngoài ra, chương trình còn có thể phát hiện các cảnh cắt trong video và trích xuất frame sau mỗi lần cắt.

## Tính năng

- Trích xuất frame từ video theo tỷ lệ phần trăm được chỉ định
- Phát hiện cảnh cắt trong video bằng phương pháp cơ bản hoặc mô hình Nova Pro
- Phân tích frame bằng AWS Bedrock với mô hình Claude
- Giao diện web thân thiện với người dùng sử dụng Streamlit
- Tự động tạo thư mục đầu ra nếu chưa tồn tại
- Hiển thị thông tin về video (tổng số frame, FPS)
- Hiển thị tiến trình trích xuất
- Lưu các frame dưới dạng file JPG với tên được đánh số

## Yêu cầu

- Python 3.7 or higher
- OpenCV (cv2)
- NumPy
- tqdm
- PyTorch (cho phương pháp Nova Pro)
- Transformers (cho phương pháp Nova Pro)
- Boto3 (cho AWS Bedrock)
- Streamlit (cho giao diện web)

Cài đặt thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## Cách sử dụng

### Giao diện dòng lệnh

#### Trích xuất frame theo tỷ lệ phần trăm

```bash
python frame_extractor.py đường_dẫn_video thư_mục_lưu_frame tỷ_lệ_phần_trăm
```

#### Phát hiện cảnh cắt và trích xuất frame

```bash
python scene_detector.py đường_dẫn_video thư_mục_lưu_frame [--method phương_pháp] [--threshold ngưỡng] [--min_scene_length độ_dài_tối_thiểu]
```

#### Sử dụng giao diện thống nhất

```bash
python main.py extract đường_dẫn_video thư_mục_lưu_frame tỷ_lệ_phần_trăm [--analyze]
```

```bash
python main.py detect đường_dẫn_video thư_mục_lưu_frame [--method phương_pháp] [--threshold ngưỡng] [--min_scene_length độ_dài_tối_thiểu]
```

### Giao diện web với Streamlit

```bash
streamlit run streamlit_app.py
```

Sau khi chạy lệnh trên, giao diện web sẽ được mở trong trình duyệt của bạn. Tại đây, bạn có thể:
1. Tải lên video
2. Chọn chế độ trích xuất frame hoặc phát hiện cảnh cắt
3. Điều chỉnh các tham số
4. Xem kết quả trực quan

## Tham số

### Trích xuất frame
- `đường_dẫn_video`: Đường dẫn đến file video cần trích xuất frame
- `thư_mục_lưu_frame`: Thư mục để lưu các frame được trích xuất
- `tỷ_lệ_phần_trăm`: Tỷ lệ phần trăm số frame cần trích xuất (từ 0 đến 100)
- `--analyze`: Phân tích các frame đã trích xuất bằng AWS Bedrock

### Phát hiện cảnh cắt
- `đường_dẫn_video`: Đường dẫn đến file video cần phát hiện cảnh cắt
- `thư_mục_lưu_frame`: Thư mục để lưu các frame được trích xuất
- `--method`: Phương pháp phát hiện, có thể là "basic" (nhanh hơn), "nova_pro" (chính xác hơn) hoặc "bedrock" (sử dụng AWS)
- `--threshold`: Ngưỡng phát hiện cảnh cắt (0-255 cho phương pháp basic, 0-1 cho phương pháp nova_pro và bedrock)
- `--min_scene_length`: Số frame tối thiểu giữa các cảnh cắt
- `--sample_rate`: Xử lý mỗi n frame (chỉ cho phương pháp bedrock)

## Cấu hình AWS

Để sử dụng các tính năng AWS Bedrock:

1. Cài đặt AWS CLI:
   ```bash
   pip install awscli
   ```

2. Cấu hình thông tin đăng nhập AWS:
   ```bash
   aws configure
   ```
   
   Bạn cần cung cấp:
   - AWS Access Key ID
   - AWS Secret Access Key
   - Default region name (ví dụ: us-east-1)
   - Default output format (json)

3. Đảm bảo tài khoản AWS của bạn có quyền truy cập vào Amazon Bedrock và mô hình Claude.

## Cách hoạt động

### Trích xuất frame theo tỷ lệ phần trăm
1. Đọc file video và xác định tổng số frame
2. Tính toán số frame cần trích xuất dựa trên tỷ lệ phần trăm
3. Tính khoảng cách giữa các frame cần trích xuất để đảm bảo phân bố đều
4. Trích xuất và lưu các frame theo khoảng cách đã tính
5. Hiển thị tiến trình và thông báo khi hoàn thành

### Phát hiện cảnh cắt
1. Đọc file video và xử lý từng frame
2. Phương pháp cơ bản: So sánh sự khác biệt giữa các frame liên tiếp
3. Phương pháp Nova Pro: Sử dụng mô hình Nova Pro để tạo embedding cho mỗi frame và so sánh sự tương đồng
4. Phương pháp Bedrock: Sử dụng AWS Bedrock với mô hình Claude để phân tích nội dung của frame
5. Khi phát hiện cảnh cắt, lưu frame sau cảnh cắt
6. Hiển thị tiến trình và thông báo khi hoàn thành

## Định dạng đầu ra

- Trích xuất frame theo tỷ lệ: Các frame được lưu dưới dạng file JPG với tên theo định dạng `frame_XXXX.jpg`
- Phát hiện cảnh cắt: Các frame được lưu dưới dạng file JPG với tên theo định dạng `scene_XXXX.jpg`
- Phân tích AWS Bedrock: Kết quả phân tích được lưu dưới dạng file JSON

## Ứng dụng

- Tạo bộ dữ liệu hình ảnh từ video
- Phân tích nội dung video
- Tạo thumbnails hoặc preview cho video
- Xử lý và phân tích hình ảnh từ video
- Phát hiện và phân đoạn cảnh trong video