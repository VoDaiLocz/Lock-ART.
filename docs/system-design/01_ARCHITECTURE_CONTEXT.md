# 01. Architecture Context

## Problem Statement

AuraLock giải quyết bài toán tạo biến thể ảnh giảm khả năng bị mô hình thị giác
học bắt chước style/subject, đồng thời giữ chất lượng đủ dùng cho công bố.

## System Boundary

Trong phạm vi:

- Tiền xử lý ảnh.
- Sinh ảnh đã bảo vệ theo profile.
- Tính metric và xuất báo cáo JSON.
- Lập benchmark manifest (LoRA / Anti-DreamBooth).

Ngoài phạm vi:

- Huấn luyện mô hình nền quy mô lớn.
- Dịch vụ online nhiều tenant production-grade.

## Primary Actors

- Artist/Researcher: chạy CLI để bảo vệ ảnh.
- Evaluator: đọc report, so profile.
- Maintainer: tối ưu profile + pipeline.
