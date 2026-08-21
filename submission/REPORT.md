# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Đỗ Khải Hoàn
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

| Run         | Vị trí      |             r | Trainable params |      LR | Train loss (NB4) | Train time (s) | Peak VRAM (GB) |
| ----------- | ----------- | ------------: | ---------------: | ------: | ---------------: | -------------: | -------------: |
| `correct`   | text-linear |            16 |       32,464,896 |  0.0001 |       **0.6266** |      **987.0** |      **12.01** |
| `attn_only` | q,v only    | 283 (matched) |       32,456,704 |  0.0001 |       **0.5371** |      **816.8** |      **12.02** |
| `wrong_lr`  | text-linear |            16 |       32,464,896 | 0.00001 |       **1.5704** |      **964.5** |      **12.01** |
| `qlora`     | text-linear |            16 |       32,464,896 |  0.0001 |       **0.7058** |     **1026.3** |       **7.09** |

### 4.1 — `attn_only` và `correct`

`attn_only` chỉ áp dụng LoRA lên các projection `q,v` và sử dụng rank `r=283` để số lượng tham số trainable gần bằng cấu hình `correct`.

Hai cấu hình có số tham số gần như tương đương:

* `correct`: **32,464,896** trainable parameters.
* `attn_only`: **32,456,704** trainable parameters.

Tuy nhiên, `attn_only` đạt train loss **0.5371**, thấp hơn `correct` ở **0.6266**.

Điều này cho thấy trong thí nghiệm NB4, attention-only với rank được tăng rất lớn có khả năng fit tập training tốt hơn. Tuy nhiên, đây **không phải là so sánh chỉ dựa trên rank**, vì `attn_only` sử dụng `r=283` trong khi `correct` chỉ sử dụng `r=16`.

Đáng chú ý, `attn_only` cũng train nhanh hơn: **816.8 s** so với **987.0 s**, trong khi peak VRAM gần như giống nhau (**12.02 GB** và **12.01 GB**).

### 4.2 — `wrong_lr`

`wrong_lr` sử dụng cùng placement và rank với `correct` nhưng learning rate giảm từ **1e-4 xuống 1e-5**, tức thấp hơn **10 lần**.

Kết quả:

* `correct`: train loss **0.6266**
* `wrong_lr`: train loss **1.5704**

Trong quá trình training, `wrong_lr` chỉ đạt mean token accuracy khoảng **79.05%** ở cuối epoch 2, trong khi `correct` đạt khoảng **99.50%**.

Kết quả này cho thấy learning rate quá thấp khiến LoRA cập nhật chậm và không đạt mức hội tụ tốt trong cùng ngân sách training.

Vì vậy, `wrong_lr` là một cấu hình không phù hợp trong thí nghiệm này.

### 4.3 — `qlora`

`qlora` sử dụng LoRA trên toàn bộ text-linear layers với `r=16` và learning rate `1e-4`, nhưng model được quantize xuống **4-bit**.

Kết quả train loss cuối:

* `correct`: **0.6266**
* `qlora`: **0.7058**

QLoRA đạt loss cao hơn một chút so với LoRA 16-bit. Tuy nhiên, ưu điểm rõ ràng nhất nằm ở bộ nhớ GPU:

* LoRA 16-bit: **12.01 GB VRAM**
* QLoRA 4-bit: **7.09 GB VRAM**

Như vậy, QLoRA giảm khoảng **4.92 GB VRAM**, tương đương khoảng **41%** so với cấu hình `correct`.

Đổi lại, thời gian training tăng:

* `correct`: **987.0 s**
* `qlora`: **1026.3 s**

Trong thí nghiệm NB4 này, QLoRA vì vậy đánh đổi **VRAM thấp hơn** để lấy **thời gian training dài hơn** và train loss cao hơn.

### 4.4 — Kết luận

Kết quả thực tế của NB4 cho thấy:

1. **`attn_only`** đạt train loss thấp nhất (**0.5371**) nhưng phải tăng rank lên **283** để match số lượng trainable parameters với `correct`.
2. **`wrong_lr`** cho kết quả kém nhất với train loss **1.5704**, cho thấy learning rate thấp hơn 10 lần làm quá trình thích nghi của LoRA kém hiệu quả.
3. **`qlora`** đạt train loss **0.7058**, cao hơn `correct`, nhưng giảm peak VRAM từ **12.01 GB xuống 7.09 GB**.
4. `correct` có train loss **0.6266**, thời gian **987.0 s** và VRAM **12.01 GB**, là baseline phù hợp để so sánh các cấu hình còn lại.

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
