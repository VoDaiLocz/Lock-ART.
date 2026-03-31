# 02. Architecture Principles

1. **Reproducibility first**: luôn ưu tiên khả năng tái lập hơn số điểm cao nhất.
2. **Transparent trade-offs**: mọi profile phải nêu rõ đổi chác quality/protection.
3. **CPU-first baseline**: workflow cơ bản chạy được không cần GPU.
4. **Benchmark-driven changes**: thay đổi logic phải đi kèm bằng chứng đo lường.
5. **Small composable modules**: tách attack/core/service/ui rõ ràng.
6. **Fail loudly in automation**: adaptive mode có thể trả non-zero khi dưới ngưỡng.
7. **Security-aware defaults**: tránh nhận input nguy hiểm, hạn chế path traversal.
