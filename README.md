# 🛡️ Ứng dụng Phát hiện Giao dịch Gian lận bằng Học máy

Ứng dụng web được xây dựng dựa trên mã nguồn huấn luyện mô hình phân loại **Random Forest** giúp tự động hóa quá trình phân tích dữ liệu, đánh giá độ chính xác, và đưa ra quyết định chẩn đoán giao dịch có hành vi gian lận hoặc rủi ro tín dụng cao.

## 🚀 Tính năng chính của Hệ thống
- **Tải dữ liệu linh hoạt:** Hỗ trợ nạp trực tiếp tập dữ liệu huấn luyện mẫu (định dạng `.csv`, `.xlsx`).
- **Phân tích tổng quan & Trực quan hóa:** Khám phá cấu trúc bảng dữ liệu, phân phối của các biến đầu vào ảnh hưởng trực tiếp đến mô hình (`X_1` đến `X_14`) và nhãn mục tiêu (`default`).
- **Kiểm định trực quan:** Đánh giá độ tin cậy của thuật toán qua Ma trận nhầm lẫn (Confusion Matrix), điểm F1-Score, Độ nhạy (Recall), và Báo cáo phân loại chi tiết.
- **Sử dụng thực tế (Inference):** Cho phép nhập thông số thủ công của một giao dịch đơn lẻ hoặc tải lên danh sách chuỗi giao dịch lớn để quét cảnh báo hàng loạt rồi xuất báo cáo.

## 📂 Cấu trúc schema dữ liệu đầu vào kỳ vọng
Để ứng dụng hoạt động chính xác, tệp dữ liệu tải lên cần tuân thủ cấu trúc định dạng bao gồm các biến:
- `X_1` đến `X_14`: Các thuộc tính số liên tục biểu diễn thông tin mã hóa của giao dịch.
- `default`: Biến mục tiêu phân loại nhị phân nhận giá trị (`0`: Hợp lệ, `1`: Gian lận/Rủi ro).

## 🛠️ Hướng dẫn cài đặt và Chạy ứng dụng

### Bước 1: Chuẩn bị môi trường máy tính
Đảm bảo bạn đã cài đặt phiên bản Python ổn định (Khuyến nghị phiên bản `Python 3.10` hoặc `3.12`).

### Bước 2: Cài đặt các thư viện phụ thuộc bắt buộc
Mở Terminal/Command Prompt tại thư mục chứa mã nguồn dự án và thực thi lệnh sau:
```bash
pip install -r requirements.txt
