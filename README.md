# Ship Detection

Bộ khung thực nghiệm phát hiện tàu trên ảnh SAR/vệ tinh với **oriented bounding box (OBB)**.

## Cấu trúc

```text
ship_detection/
├── configs/                  # Cấu hình chung và manifest của từng mô hình
├── data/
│   ├── raw/                  # Dữ liệu gốc, chỉ đọc
│   └── processed/            # DOTA/MMRotate: train, val, test
├── docs/                     # Quy ước dữ liệu và thiết kế benchmark
├── notebooks/                # Notebook EDA/phân tích; không chứa pipeline chính
├── outputs/                  # Checkpoint, log và dự đoán (không commit)
├── requirements/             # Môi trường tách riêng Ultralytics/MMRotate
├── scripts/                  # Công cụ dòng lệnh dùng chung
├── src/ship_detection/       # Mã nguồn dùng lại
└── tests/                    # Kiểm thử nhanh
```



## Quy trình benchmark

1. Khóa một lần chia `train/val/test` dùng chung cho cả ba mô hình.
2. Giữ nguyên ảnh và annotation OBB; chỉ chuyển định dạng ở adapter của từng framework.
3. Huấn luyện theo manifest trong `configs/models/` và ghi mọi artifact vào `outputs/<model>/<run_id>/`.
4. Chọn checkpoint trên `val`; chỉ báo cáo một lần trên `test`.
5. So sánh OBB mAP50, mAP50-95, precision, recall, F1, latency, VRAM và số tham số.
