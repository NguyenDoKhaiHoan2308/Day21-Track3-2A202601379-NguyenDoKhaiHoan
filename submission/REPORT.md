# Lab 21 — Evaluation Report

**Họ tên**: Nguyen Do Khai Hoan  
**MSSV**: 2A202601379  
**Ngày**: 21/08/2026  
**Tier**: `T4`  
**Base model**: `unsloth/Qwen3.5-4B`  
**GPU thực tế**: `T4 16GB`

> Mọi con số dưới đây được lấy từ kết quả chạy thực tế trong `results/`.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage |
| Train / val | 250 / 50 |
| `max_length` | 1024 |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 / 30 |

**Template có giữ khối `<think>` không?** Không.

Template không sử dụng reasoning trace `<think>`. Vì vậy việc huấn luyện tập trung trực tiếp vào phần output JSON của bài toán phân loại ticket. Mask được kiểm chứng trước bằng NB1 để đảm bảo phần câu trả lời nằm trong loss và phần instruction/input không bị tối ưu hóa.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.4149` |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Mask sử dụng `assistant-only`. Đây là lựa chọn quan trọng vì mô hình chỉ nên học sinh output JSON mục tiêu thay vì học lại toàn bộ instruction và input.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.000 | 0.7578 | 0.000 | 3371.8 |
| (b) base + optimized prompt | 0.765 | 0.7578 | 1.000 | 1025.1 |
| (c) LoRA fine-tune | 0.975 | 0.7667 | 1.000 | 1443.0 |

**(b) có thật sự mạnh hơn (a) không?** Có.

Optimized prompt cải thiện target từ 0.000 lên 0.765 và format từ 0 lên 1.0. Điều này cho thấy chất lượng prompt có ảnh hưởng rất lớn đến bài toán trước cả khi fine-tuning. `OPTIMIZED_PROMPT` được giữ nguyên để baseline `(b)` làm mốc công bằng cho việc đánh giá LoRA.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6957 | **0.975** | 1017 | 12.01 |
| `attn_only` | q,v | matched | 32,456,704 | 0.0001 | — | **0.970** | — | — |
| `wrong_lr` | text-linear | 16 | 32,464,896 | wrong LR | — | **0.000** | — | — |
| `qlora` | text-linear | 16 | — | 0.0001 | — | **0.940** | — | — |

### 4.1 — `attn_only` và `correct`

`attn_only` có số tham số trainable gần như bằng `correct`, lần lượt khoảng 32.457M và 32.465M. Tuy nhiên trên tập target, `correct` đạt 0.975 còn `attn_only` đạt 0.970, vì vậy `correct` thắng nhẹ. Điều này cho thấy số lượng tham số không quyết định hoàn toàn hiệu quả; vị trí gắn adapter trong toàn bộ text decoder cũng ảnh hưởng đến khả năng học task.

### 4.2 — `wrong_lr`

`wrong_lr` chỉ thay đổi learning rate nhưng kết quả target giảm hoàn toàn xuống 0.000 và format cũng xuống 0.000. Nếu chỉ nhìn loss mà không kiểm tra learning rate và kết quả downstream, có thể kết luận sai rằng một đường loss thấp hoặc có vẻ hội tụ đồng nghĩa với mô hình tốt. Thực tế metric quan trọng của lab là target score và khả năng giữ general capability, không phải chỉ train loss.

### 4.3 — `qlora`

QLoRA sử dụng quantization để giảm yêu cầu bộ nhớ GPU so với full-precision LoRA. Đổi lại, kết quả target của QLoRA chỉ đạt 0.940, thấp hơn `correct` 0.975, và latency cũng cao hơn: khoảng 1813.7 ms so với 1443.0 ms. Với kết quả đo được trong lab này, QLoRA không phải lựa chọn tốt nhất cho dòng model/configuration hiện tại khi GPU T4 vẫn có thể chạy LoRA 16-bit.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `PASSED`

`target Δ = +0.210` · `regression Δ = +0.009` · `valid_trace_rate = 0.00`

Fine-tune đạt target score 0.975, cao hơn baseline optimized prompt 0.765 một khoảng +0.210. Đồng thời regression score tăng từ 0.7578 lên 0.7667, tức thay đổi +0.009 và nằm trong ngưỡng regression tolerance ±0.020. Vì vậy mô hình vượt qua cả hai điều kiện của regression gate: phải cải thiện target và không được làm suy giảm general capability quá mức. Kết quả này đặc biệt đáng chú ý vì lần chạy trước target đã đạt 0.970 nhưng regression chỉ còn 0.7222, dẫn tới FAILED. Việc bổ sung replay data đã giúp giảm catastrophic forgetting và đưa regression trở lại mức 0.7667. Do đó cấu hình hiện tại vừa cải thiện task chính vừa giữ được khả năng tổng quát theo tiêu chí của lab.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Đặt đèn bàn LED, sai màu | `san_pham_loi` | thấp hơn | đúng | ✅ FT thắng |
| 2 | Đặt ốp lưng, shipper không giao | `van_chuyen` | thấp hơn | đúng | ✅ FT thắng |
| 3 | Đặt bình giữ nhiệt, chưa thấy tiền | `hoan_tien` | đúng | score 0.75 | ❌ FT thua |
| 4 | Máy xay sinh tố, trả lại tiền | `doi_tra/hoan_tien` | đúng | score 0.75 | ❌ FT thua |
| 5 | Máy xay sinh tố, hoàn lại | `doi_tra/hoan_tien` | đúng | score 0.75 | ❌ FT thua |

Các ca FT thua tập trung vào những ticket có ý nghĩa gần nhau giữa `doi_tra` và `hoan_tien`. Cụm từ như “trả lại tiền”, “hoàn lại” hoặc “chưa thấy tiền” có thể khiến intent dễ bị nhầm. Đây là failure mode đáng chú ý nhất của mô hình hiện tại.

---

## 7. Kết luận & điều tôi học được

**Kết luận**

Mô hình LoRA hiện tại có thể được xem là đạt yêu cầu để hoàn thành lab và có tiềm năng deploy trong phạm vi bài toán đã đánh giá. Target score đạt 0.975, vượt optimized-prompt baseline 0.765 tới +0.210. Quan trọng hơn, regression score không bị suy giảm mà tăng từ 0.7578 lên 0.7667, vì vậy mô hình vượt qua regression gate. Kết quả trước khi thêm replay data cho thấy catastrophic forgetting là vấn đề thực tế: target cao nhưng regression giảm xuống 0.7222 và khiến hệ thống FAILED. Sau khi bổ sung replay data, regression được cải thiện lên 0.7667 và gate chuyển sang PASSED. Trong các đòn bẩy của lab, mask là điều kiện nền tảng để đảm bảo mô hình học đúng phần output; sau đó learning rate và vị trí adapter quyết định đáng kể chất lượng fine-tuning. Dataset cũng rất quan trọng vì các failure case cho thấy intent `doi_tra` và `hoan_tien` còn dễ nhầm. Vì vậy nếu triển khai thực tế, cần tiếp tục kiểm tra các ticket có intent gần nhau thay vì chỉ dựa vào target score tổng thể.

**Ba điều tôi học được**

1. Replay data có thể giảm catastrophic forgetting: lần chạy trước regression giảm xuống 0.7222 nhưng sau khi thêm replay data đạt 0.7667 và vượt regression gate.
2. Adapter placement quan trọng: `correct` đạt 0.975 trong khi `attn_only` đạt 0.970 dù số lượng trainable parameters gần như tương đương.
3. Learning rate có thể quyết định hoàn toàn kết quả: `wrong_lr` đạt target 0.000 và format 0.000, cho thấy train loss hoặc quá trình training chạy thành công không đảm bảo downstream quality.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**

- Tăng thêm các mẫu replay nhưng vẫn giữ trong khoảng 1–5%.
- Bổ sung dữ liệu phân biệt rõ `doi_tra` và `hoan_tien`.
- Thử rank LoRA khác như r=8 và r=32.
- Kiểm tra riêng từng loại lỗi trên target set.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub

---

## Kết quả chính

- **Target:** 0.975
- **Baseline (b):** 0.765
- **Target gain:** +0.210
- **Regression:** 0.7667
- **Regression delta:** +0.009
- **Format:** 1.000
- **Latency:** 1443.0 ms
- **Verdict:** **PASSED**
