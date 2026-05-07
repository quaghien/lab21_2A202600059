# REPORT.md

## 1. Setup

- **Submission option**: Option B - GitHub + HuggingFace Hub
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Fine-tuning method**: QLoRA 4-bit + LoRA adapters with Unsloth + TRL `SFTTrainer`
- **Notebook used**: `notebooks/Lab21_LoRA_Finetuning_L4.ipynb`
- **Actual GPU runtime**: `NVIDIA L4` with `22.5 GB VRAM`
- **CUDA / PyTorch**: CUDA `12.8`, PyTorch `2.10.0+cu128`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`
- **Dataset size used**: 200 samples
- **Train / eval split**: 180 / 20
- **Detected text columns**: `instruction_vi`, `input_vi`, `output_vi`
- **Token length analysis**: `p50=227`, `p95=562`, `p99=704`, so `max_seq_length=1024`
- **Training config**: 3 epochs, learning rate `2e-4`, cosine scheduler, `warmup_ratio=0.10`, batch size per device `4`, gradient accumulation `8`, effective batch size `32`
- **Estimated training cost**: total training time about `4.4 minutes`, estimated cost about `$0.03` at `$0.35/hour`

Ghi chú: notebook gốc là biến thể tối ưu cho model 3B, nhưng phiên chạy thực tế trong output là trên GPU `NVIDIA L4`, nên kết quả cuối cùng phản ánh điều kiện L4 chứ không phải T4.

## 2. Rank Experiment Results

| Rank | Alpha | Trainable params | Train time (min) | Peak VRAM (GB) | Eval loss | Eval perplexity |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 1,843,200 | 1.4119 | 13.9105 | 1.7410 | 5.7031 |
| 16 | 32 | 3,686,400 | 1.5394 | 13.3131 | 1.6952 | 5.4478 |
| 64 | 128 | 14,745,600 | 1.4232 | 14.7129 | 1.5838 | 4.8733 |

Nhận xét nhanh:

- `r=64` cho perplexity tốt nhất, giảm từ `5.7031` ở `r=8` xuống `4.8733`.
- `r=16` là điểm cân bằng khá tốt: perplexity tốt hơn `r=8`, nhưng số tham số trainable vẫn thấp hơn rất nhiều so với `r=64`.
- VRAM không tăng quá mạnh giữa ba cấu hình do base model đã được nạp 4-bit, nhưng `r=64` vẫn là mức dùng bộ nhớ cao nhất.
- Training time giữa ba rank gần như tương đương trong bài lab này vì dataset nhỏ, nên khác biệt chủ yếu nằm ở chất lượng và số tham số trainable.

## 3. Loss Curve Analysis

Từ file `results/loss_curve.png` và log trainer, cả ba rank đều có train loss giảm khá ổn định trong 3 mốc log. Không có dấu hiệu diverge hay bùng nổ loss. Vì notebook này tắt `eval-during-training` để tiết kiệm VRAM, mình không có đường `eval loss` theo từng bước để kết luận overfitting một cách mạnh. Tuy vậy, dựa trên eval cuối cùng:

- `r=8` có chất lượng thấp nhất, cho thấy rank này hơi thiếu capacity.
- `r=16` cải thiện nhẹ và ổn định.
- `r=64` tiếp tục cải thiện perplexity mà chưa xuất hiện dấu hiệu suy giảm trên tập eval.

Với dataset chỉ 200 mẫu và tổng số bước huấn luyện chỉ 18, mình đánh giá chưa có dấu hiệu overfitting rõ rệt trong run này. Nếu train thêm nhiều epoch hơn hoặc tăng rank tiếp, khi đó nên bật eval định kỳ trên L4 để theo dõi khoảng cách giữa train loss và eval loss chặt hơn.

## 4. Qualitative Comparison

So sánh base model với adapter `r=16` trên 5 prompt trong `results/qualitative_comparison.csv`:

| Prompt | Base model | Fine-tuned r=16 | Nhận xét |
|---|---|---|---|
| Giải thích machine learning cho người mới bắt đầu | Câu trả lời đúng ý nhưng diễn đạt ngắn và hơi chung chung | Giải thích mạch lạc hơn, tiếng Việt tự nhiên hơn | Fine-tuned tốt hơn |
| Viết code Fibonacci | Base trả lời bị lẫn `hello_world()` không liên quan trước khi vào lời giải | Fine-tuned vào đúng bài toán hơn | Fine-tuned tốt hơn rõ |
| Liệt kê 5 nguyên tắc UI/UX | Base bị trộn tiếng Trung trong câu trả lời | Fine-tuned trả lời bằng tiếng Việt nhất quán hơn | Fine-tuned thắng rất rõ |
| Tóm tắt LoRA vs QLoRA | Cả hai đều trả lời đúng hướng nhưng còn hơi khái quát | Fine-tuned dùng thuật ngữ đúng hơn và rõ hơn đôi chút | Fine-tuned nhỉnh hơn |
| Phân biệt prompt engineering, RAG, fine-tuning | Base đúng ý nhưng diễn đạt còn mơ hồ | Fine-tuned phân biệt vai trò từng kỹ thuật rõ hơn | Fine-tuned tốt hơn |

Kết luận qualitative: adapter `r=16` cải thiện độ ổn định ngôn ngữ tiếng Việt, giảm hiện tượng lẫn ngôn ngữ ngoài ý muốn, và bám sát instruction tốt hơn so với base model. Đây là tín hiệu rất tích cực vì dataset dùng cho lab khá nhỏ.

## 5. Conclusion về Rank Trade-off

Trong thí nghiệm này, mình xem `r=16` là lựa chọn thực dụng tốt nhất, còn `r=64` là lựa chọn tốt nhất nếu ưu tiên chất lượng thuần túy. Lý do là `r=64` cho perplexity thấp nhất (`4.8733`) và không làm training time tăng đáng kể trong bối cảnh dataset chỉ có 200 mẫu. Tuy nhiên, số tham số trainable của `r=64` tăng lên `14,745,600`, gấp 4 lần `r=16` và gấp 8 lần `r=8`, đồng thời peak VRAM cũng cao nhất. Với một bài lab có mục tiêu cân bằng giữa hiệu quả, chi phí và khả năng tái lập, `r=16` hợp lý hơn vì nó giữ được footprint nhỏ, perplexity đã cải thiện rõ so với `r=8`, và qualitative output cũng tốt hơn base model một cách ổn định. Ngược lại, `r=8` hơi thiếu capacity nên chất lượng thấp nhất. Vì vậy, nếu triển khai thực tế với ràng buộc tài nguyên hoặc cần workflow production gọn nhẹ, mình sẽ chọn `r=16`; còn nếu ưu tiên tối đa chất lượng và vẫn đủ VRAM, `r=64` là ứng viên mạnh hơn.

## 6. What I Learned

- QLoRA 4-bit giúp fine-tune một mô hình 3B khá thoải mái trên GPU `NVIDIA L4`, trong khi mức VRAM đỉnh vẫn chỉ quanh 13-15 GB.
- Rank cao hơn không phải lúc nào cũng làm thời gian train tăng nhiều, nhất là khi dataset nhỏ; trade-off lớn hơn thường nằm ở số tham số trainable và bộ nhớ.
- Chỉ với 200 mẫu tiếng Việt, adapter vẫn có thể cải thiện instruction-following và độ nhất quán ngôn ngữ khá rõ so với base model.
