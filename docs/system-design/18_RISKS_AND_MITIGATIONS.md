# 18. Risks and Mitigations

## R1: Over-claiming effectiveness

- Mitigation: bắt buộc mục limitations trong report/docs.

## R2: Metric drift giữa phiên bản

- Mitigation: version hóa schema và bộ ảnh chuẩn.

## R3: Dependency breakage

- Mitigation: lock file cho benchmark quan trọng.

## R4: Resource exhaustion khi batch lớn

- Mitigation: chunking, retry, và giới hạn kích thước input.
