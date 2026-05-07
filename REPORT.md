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

Ghi chú kỹ thuật:

- Notebook đặt tên theo workflow cho model 3B, nhưng output thực tế xác nhận phiên chạy trên `NVIDIA L4`, nên các chỉ số trong report phản ánh môi trường L4 chứ không phải T4.
- Base model dùng quantization 4-bit, còn phần train chỉ cập nhật LoRA adapters. Đây là lý do peak VRAM thực tế thấp hơn khá nhiều so với việc fine-tune toàn bộ trọng số.
- Dataset chỉ có `200` mẫu nên đây là bài toán fine-tune quy mô nhỏ. Vì vậy, kết quả rank experiment cần được hiểu là trade-off trên một low-data regime, chưa phải kết luận tuyệt đối cho production quy mô lớn.

Ý nghĩa của setup này là: mô hình nền đủ nhỏ để thử nghiệm nhanh, dataset đủ gọn để so sánh rank trong thời gian ngắn, và môi trường L4 22.5 GB cho phép chạy QLoRA khá thoải mái mà chưa cần tối ưu quá cực đoan. Điều này rất phù hợp với mục tiêu của lab là hiểu cơ chế rank trade-off thay vì chỉ tối ưu benchmark.

## 2. Rank Experiment Results

| Rank | Alpha | Trainable params | Train time (min) | Peak VRAM (GB) | Eval loss | Eval perplexity |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 1,843,200 | 1.4119 | 13.9105 | 1.7410 | 5.7031 |
| 16 | 32 | 3,686,400 | 1.5394 | 13.3131 | 1.6952 | 5.4478 |
| 64 | 128 | 14,745,600 | 1.4232 | 14.7129 | 1.5838 | 4.8733 |

Phân tích kỹ thuật:

- `r=16` cải thiện perplexity khoảng `4.48%` so với `r=8`, trong khi số tham số trainable tăng đúng `2x`.
- `r=64` cải thiện perplexity khoảng `14.55%` so với `r=8`, nhưng số tham số trainable tăng tới `8x`.
- Từ `r=16` lên `r=64`, perplexity tiếp tục giảm từ `5.4478` xuống `4.8733`, tức cải thiện thêm khoảng `10.55%` tương đối so với mốc `r=16`.
- Peak VRAM thay đổi ít hơn nhiều so với số tham số trainable. `r=64` chỉ cao hơn `r=8` khoảng `0.80 GB`, cho thấy trong bối cảnh QLoRA 4-bit, bộ nhớ của base model vẫn chiếm phần lớn footprint.
- Training time gần như không tách bạch giữa ba rank. Chênh lệch lớn nhất giữa `r=8` và `r=16` chỉ khoảng `7.65 giây`, còn `r=64` gần như tương đương `r=8`. Điều này xảy ra vì dataset nhỏ, tổng số bước huấn luyện chỉ `18`, nên compute overhead do rank tăng chưa đủ lớn để bộc lộ rõ.

Nhìn dưới góc độ thực nghiệm, đây là một kết quả thú vị: khi dữ liệu nhỏ và số bước ít, rank tác động mạnh nhất lên chất lượng đầu ra và số tham số cập nhật, trong khi ảnh hưởng lên thời gian huấn luyện có thể bị “nén lại”. Vì vậy, nếu chỉ nhìn train time thì rất dễ kết luận sai rằng tăng rank “gần như miễn phí”, trong khi thực tế chi phí tiềm ẩn nằm ở adapter size, khả năng scale sang dataset lớn hơn, và nguy cơ over-parameterization.

Diminishing returns trong thí nghiệm này chưa xuất hiện rõ. Từ `r=8` sang `r=16` có cải thiện, và từ `r=16` sang `r=64` vẫn tiếp tục cải thiện. Tuy nhiên, do tập eval chỉ có `20` mẫu, mình xem đây là tín hiệu tích cực chứ chưa coi là bằng chứng đủ mạnh để kết luận `r=64` sẽ luôn tốt hơn trong mọi lần chạy.

## 3. Loss Curve Analysis

Từ file `results/loss_curve.png` và log trainer, cả ba rank đều có train loss giảm khá ổn định trong 3 mốc log. Không có dấu hiệu diverge, loss spike, hay training instability. Đây là một tín hiệu quan trọng vì với rank cao hơn, đặc biệt `r=64`, đôi khi có thể gặp dao động lớn hơn nếu learning rate quá mạnh hoặc dữ liệu nhiễu. Trong run này điều đó chưa xảy ra.

Vì notebook tắt `eval-during-training` để tiết kiệm VRAM, mình không có chuỗi `eval loss` theo từng bước để kiểm tra trực tiếp khoảng cách giữa train và eval. Do đó, phần kết luận về overfitting phải dựa trên sự kết hợp giữa:

- xu hướng train loss giảm mượt,
- eval loss cuối cùng của từng rank,
- kích thước dataset nhỏ,
- và chất lượng qualitative đầu ra.

Dựa trên các tín hiệu đó:

- `r=8` có chất lượng thấp nhất, cho thấy rank này hơi thiếu capacity.
- `r=16` cải thiện nhẹ và ổn định.
- `r=64` tiếp tục cải thiện perplexity mà chưa xuất hiện dấu hiệu suy giảm trên tập eval.

Với dataset chỉ `200` mẫu và tổng số bước huấn luyện chỉ `18`, mình đánh giá chưa có dấu hiệu overfitting rõ rệt trong run hiện tại. Tuy nhiên, đây chưa phải là bằng chứng rằng overfitting không tồn tại. Thực ra với low-data setup, rank cao như `r=64` có thể học rất nhanh cả tín hiệu tốt lẫn nhiễu. Việc chưa thấy overfitting nhiều khả năng đến từ việc training khá ngắn và eval set cũng nhỏ.

Nếu muốn đánh giá sâu hơn ở góc nhìn kỹ thuật, lần chạy sau nên:

- bật eval định kỳ trên L4 vì VRAM vẫn còn đủ biên,
- log thêm base-model perplexity để đo mức cải thiện tuyệt đối,
- tăng số mẫu eval hoặc chạy nhiều random seed để giảm phương sai phép đo.

## 4. Qualitative Comparison

So sánh base model với adapter `r=16` trên 5 prompt trong `results/qualitative_comparison.csv`:

| Prompt | Base model | Fine-tuned r=16 | Nhận xét |
|---|---|---|---|
| Giải thích machine learning cho người mới bắt đầu | Câu trả lời đúng ý nhưng diễn đạt ngắn và hơi chung chung | Giải thích mạch lạc hơn, tiếng Việt tự nhiên hơn | Fine-tuned tốt hơn |
| Viết code Fibonacci | Base trả lời bị lẫn `hello_world()` không liên quan trước khi vào lời giải | Fine-tuned vào đúng bài toán hơn | Fine-tuned tốt hơn rõ |
| Liệt kê 5 nguyên tắc UI/UX | Base bị trộn tiếng Trung trong câu trả lời | Fine-tuned trả lời bằng tiếng Việt nhất quán hơn | Fine-tuned thắng rất rõ |
| Tóm tắt LoRA vs QLoRA | Cả hai đều trả lời đúng hướng nhưng còn hơi khái quát | Fine-tuned dùng thuật ngữ đúng hơn và rõ hơn đôi chút | Fine-tuned nhỉnh hơn |
| Phân biệt prompt engineering, RAG, fine-tuning | Base đúng ý nhưng diễn đạt còn mơ hồ | Fine-tuned phân biệt vai trò từng kỹ thuật rõ hơn | Fine-tuned tốt hơn |

Phân tích sâu hơn:

- Mẫu cải thiện rõ nhất không chỉ là “đúng hơn”, mà là **ổn định định dạng và ngôn ngữ hơn**. Ví dụ UI/UX prompt cho thấy base model bị trộn tiếng Trung, trong khi fine-tuned trả lời thuần tiếng Việt hơn. Điều này cho thấy adapter đã hấp thụ được bias mong muốn từ tập instruction tiếng Việt.
- Ở prompt viết code Fibonacci, adapter không chỉ trả lời đúng domain hơn mà còn giảm phần mở đầu lạc đề. Đây là dấu hiệu tốt của instruction alignment, không đơn thuần là memorization.
- Ở các prompt khái niệm như LoRA/QLoRA hay prompt engineering/RAG/fine-tuning, mức cải thiện có nhưng chưa “nhảy vọt”. Điều này hợp lý vì dataset lab không chuyên sâu vào một domain kỹ thuật hẹp, nên adapter chủ yếu cải thiện phong cách trả lời và độ nhất quán hơn là bổ sung tri thức mới.

Kết luận qualitative: adapter `r=16` cải thiện ba khía cạnh quan trọng là bám instruction, độ sạch ngôn ngữ tiếng Việt, và giảm lỗi trả lời lan man. Đây là tín hiệu tích cực vì mục tiêu của fine-tuning trong bài lab này không phải vá knowledge gap, mà là điều chỉnh hành vi sinh câu trả lời theo distribution của dữ liệu huấn luyện.

## 5. Conclusion về Rank Trade-off

Trong thí nghiệm này, mình xem `r=16` là lựa chọn có ROI tốt nhất, còn `r=64` là lựa chọn có chất lượng tốt nhất. Hai kết luận này không mâu thuẫn nhau vì chúng tối ưu cho hai mục tiêu khác nhau. Nếu chỉ nhìn perplexity, `r=64` thắng rõ ràng với `4.8733`, thấp hơn cả `r=16` (`5.4478`) và `r=8` (`5.7031`). Điều đó cho thấy khi tăng rank, adapter có thêm capacity để biểu diễn cập nhật low-rank tốt hơn cho bài toán instruction-following tiếng Việt. Tuy nhiên, cái giá phải trả là số tham số trainable của `r=64` tăng lên `14,745,600`, tức gấp `4x` so với `r=16` và gấp `8x` so với `r=8`.

Trong low-data regime của bài lab này, training time chưa phản ánh hết chi phí thật vì dataset nhỏ nên ba cấu hình đều train rất nhanh. Nếu scale cùng setup này lên dataset lớn hơn hoặc chạy nhiều epoch hơn, mình kỳ vọng chênh lệch giữa `r=16` và `r=64` sẽ bộc lộ rõ hơn ở thời gian, checkpoint size và rủi ro học cả nhiễu dữ liệu. Vì vậy, `r=16` là rank “thực dụng” hơn: đủ nhỏ để nhẹ, đủ lớn để cải thiện rõ so với `r=8`, và qualitative output đã ổn định hơn base model ở các prompt kiểm tra. `r=8` thì hơi thiếu capacity, thể hiện qua perplexity tệ nhất và chất lượng trả lời không ổn định bằng.

Về diminishing returns, mình chưa thấy dấu hiệu bão hòa rõ rệt trong run này vì `r=64` vẫn tiếp tục cải thiện. Nhưng mình không kết luận rằng cứ tăng rank là tốt hơn mãi. Với chỉ `20` mẫu eval, kết quả hiện tại nên được hiểu là một chỉ báo. Nếu triển khai production thật, mình sẽ chọn `r=16` cho phương án mặc định vì cân bằng tốt giữa chất lượng, kích thước adapter và khả năng tái lập. Còn nếu mục tiêu là tối đa hóa chất lượng trong khi vẫn đủ VRAM và chấp nhận adapter lớn hơn, `r=64` là lựa chọn đáng thử tiếp.

## 6. What I Learned

- QLoRA 4-bit thật sự hiệu quả cho môi trường GPU tầm trung như `NVIDIA L4 22.5 GB`; phần nặng nhất vẫn là base model, còn tăng rank chủ yếu làm tăng adapter capacity chứ không làm VRAM bùng nổ.
- Khi dataset nhỏ, training time có thể đánh lừa mình. Rank cao hơn chưa chắc chậm hơn rõ trong một run ngắn, nhưng chi phí thật lại nằm ở số tham số trainable, kích thước adapter và độ nhạy với overfitting khi scale lên bài toán lớn hơn.
- Fine-tuning trong bài này chủ yếu cải thiện hành vi trả lời hơn là thêm kiến thức mới: đầu ra tiếng Việt sạch hơn, ít lạc đề hơn và bám instruction tốt hơn. Điều đó làm mình hiểu rõ hơn khi nào nên dùng fine-tune thay vì kỳ vọng nó thay thế RAG.
