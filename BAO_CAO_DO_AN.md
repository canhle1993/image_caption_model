# BÁO CÁO ĐỒ ÁN CUỐI KHOÁ

## SINH MÔ TẢ ẢNH TỰ ĐỘNG SỬ DỤNG HỌC SÂU
### (Neural Image Caption Generation)

---

**Sinh viên thực hiện**: [Họ tên]  
**Lớp**: [Tên lớp]  
**Giảng viên hướng dẫn**: [Tên GVHD]  
**Ngày**: Tháng 4 năm 2026

---

## MỤC LỤC

1. [Giới thiệu](#1-giới-thiệu)
2. [Tổng quan lý thuyết](#2-tổng-quan-lý-thuyết)
3. [Dataset](#3-dataset)
4. [Kiến trúc mô hình](#4-kiến-trúc-mô-hình)
5. [Chi tiết thực nghiệm](#5-chi-tiết-thực-nghiệm)
6. [Kết quả và đánh giá](#6-kết-quả-và-đánh-giá)
7. [Phân tích và thảo luận](#7-phân-tích-và-thảo-luận)
8. [Kết luận và hướng phát triển](#8-kết-luận-và-hướng-phát-triển)
9. [Tài liệu tham khảo](#9-tài-liệu-tham-khảo)

---

## 1. GIỚI THIỆU

### 1.1 Đặt vấn đề

Sinh mô tả ảnh tự động (Image Captioning) là bài toán giao thoa giữa **Thị giác máy tính (Computer Vision)** và **Xử lý ngôn ngữ tự nhiên (Natural Language Processing)**. Mục tiêu là xây dựng một hệ thống có khả năng quan sát một hình ảnh và tự động tạo ra một câu mô tả ngắn gọn, chính xác về nội dung của ảnh đó — tương tự như cách con người nhìn và diễn đạt bằng lời.

Ứng dụng của bài toán này rất rộng rãi:
- **Hỗ trợ người khiếm thị**: Đọc nội dung ảnh thông qua màn hình âm thanh
- **Tìm kiếm ảnh**: Tìm kiếm hình ảnh theo mô tả ngôn ngữ tự nhiên
- **Mạng xã hội**: Tự động gợi ý caption cho ảnh đăng tải
- **Robot thông minh**: Giúp robot hiểu và mô tả môi trường xung quanh
- **Y tế**: Tự động mô tả hình ảnh y khoa (X-quang, MRI)

### 1.2 Mục tiêu đề tài

Đề tài này nghiên cứu và thực nghiệm ba hướng tiếp cận cho bài toán sinh mô tả ảnh:

1. **VGG16 + LSTM + Bahdanau Attention**: Mô hình cơ sở (baseline) sử dụng VGG16 làm encoder trích xuất đặc trưng ảnh
2. **ResNet-101 + LSTM + Bahdanau Attention**: Nâng cấp encoder lên ResNet-101 với đặc trưng 2048 chiều phong phú hơn
3. **CLIP ViT-B/32 + LSTM + Bahdanau Attention**: Sử dụng CLIP (Contrastive Language-Image Pretraining) — mô hình được huấn luyện trên 400 triệu cặp ảnh-văn bản, tận dụng sự căn chỉnh ngữ nghĩa vision-language

Cả ba mô hình được huấn luyện và đánh giá trên dataset **Flickr30k** với cùng tập train/val/test, đảm bảo kết quả so sánh công bằng.

### 1.3 Cấu trúc báo cáo

Báo cáo được tổ chức theo thứ tự: Lý thuyết nền → Dataset → Kiến trúc → Thực nghiệm → Kết quả → Phân tích → Kết luận.

---

## 2. TỔNG QUAN LÝ THUYẾT

### 2.1 Kiến trúc Encoder-Decoder

Phương pháp phổ biến nhất cho bài toán sinh mô tả ảnh là kiến trúc **Encoder-Decoder**:

- **Encoder**: Một mạng CNN (Convolutional Neural Network) hoặc Vision Transformer trích xuất đặc trưng không gian từ ảnh, tạo ra một biểu diễn vector/tensor đặc trưng
- **Decoder**: Một mạng RNN (thường là LSTM) nhận đặc trưng ảnh và sinh ra chuỗi từ một cách tự hồi quy (auto-regressive)

### 2.2 Cơ chế Attention — Bahdanau Attention

Cơ chế Attention (Chú ý) cho phép mô hình **tập trung vào từng vùng khác nhau của ảnh** tại mỗi bước sinh từ. Thay vì nén toàn bộ ảnh vào một vector cố định, attention giúp decoder "nhìn lại" các vùng ảnh liên quan khi sinh mỗi từ.

**Bahdanau Attention (Additive Attention)** được sử dụng trong cả ba mô hình:

```
e_i = V^T * tanh(W_h * h_{t-1} + W_f * f_i)   (score function)
α_i = softmax(e_i)                               (attention weights)
c_t = Σ α_i * f_i                               (context vector)
```

Trong đó:
- `h_{t-1}`: trạng thái ẩn LSTM bước trước
- `f_i`: đặc trưng ảnh tại vị trí không gian i (i = 1..49)
- `α_i`: trọng số chú ý (attention weight) tại vị trí i
- `c_t`: context vector — tổng hợp đặc trưng ảnh có trọng số

### 2.3 Teacher Forcing

Trong quá trình huấn luyện, kỹ thuật **Teacher Forcing** được sử dụng: thay vì dùng từ được dự đoán ở bước trước làm đầu vào cho bước tiếp theo, mô hình nhận **từ đúng (ground truth)** từ caption thực. Điều này giúp quá trình huấn luyện ổn định và hội tụ nhanh hơn.

### 2.4 Beam Search

Tại thời điểm inference, thay vì chọn từ có xác suất cao nhất tại mỗi bước (greedy search), **Beam Search** duy trì `k` ứng viên tốt nhất (beam width = 5) và mở rộng song song, từ đó tìm ra caption có tổng log-probability cao nhất.

### 2.5 Các mạng Encoder được sử dụng

#### 2.5.1 VGG16
VGG16 (Simonyan & Zisserman, 2014) là mạng CNN 16 lớp với kiến trúc đơn giản, đồng nhất. Đặc trưng được trích xuất từ lớp `conv5_3` (block5_pool) cho kích thước `(49, 512)` — 49 vị trí không gian × 512 kênh đặc trưng.

#### 2.5.2 ResNet-101
ResNet-101 (He et al., 2016) là mạng 101 lớp với kết nối tắt (residual connections) giải quyết vấn đề gradient vanishing trong mạng sâu. Đặc trưng trích xuất từ `layer4` cho kích thước `(49, 2048)` — phong phú gấp 4 lần VGG16.

#### 2.5.3 CLIP ViT-B/32
CLIP (Radford et al., 2021 — OpenAI) là mô hình được huấn luyện theo phương pháp **contrastive learning** trên 400 triệu cặp (ảnh, văn bản) từ internet. CLIP sử dụng Vision Transformer (ViT-B/32) làm encoder ảnh, chia ảnh thành các patch 32×32 và xử lý theo cơ chế self-attention. Đặc trưng đầu ra có kích thước `(49, 512)`, nhưng đặc biệt ở chỗ chúng đã mang thông tin **ngữ nghĩa ngôn ngữ-thị giác** nhờ quá trình huấn luyện contrastive với text.

### 2.6 GloVe Word Embeddings

**GloVe 300d** (Pennington et al., 2014) là các vector biểu diễn từ 300 chiều được huấn luyện trên 6 tỷ token từ Wikipedia và Gigaword. Trong đề tài này, embedding matrix được khởi tạo bằng GloVe và cho phép fine-tune trong quá trình huấn luyện, giúp mô hình tận dụng kiến thức ngôn ngữ sẵn có.

### 2.7 BLEU Score — Thước đo đánh giá

**BLEU (Bilingual Evaluation Understudy)** là thước đo phổ biến nhất để đánh giá chất lượng caption. BLEU so sánh n-gram của caption được sinh ra với caption tham chiếu (ground truth):

- **BLEU-1**: Độ chính xác unigram (1-gram)
- **BLEU-2**: Độ chính xác bigram (2-gram)  
- **BLEU-3**: Độ chính xác trigram (3-gram)
- **BLEU-4**: Độ chính xác 4-gram — thước đo chính trong so sánh

BLEU-4 được tính theo công thức `corpus_bleu` của `nltk`, đánh giá trên toàn bộ tập test (1,000 ảnh × 5 caption/ảnh).

---

## 3. DATASET

### 3.1 Flickr30k

**Flickr30k** là một trong những dataset chuẩn phổ biến nhất cho bài toán Image Captioning. Dataset được tổng hợp từ Flickr.com bởi Young et al. (2014).

| Thông số | Giá trị |
|---------|--------|
| Tổng số ảnh | 31,783 ảnh |
| Số caption | 158,915 caption (5 caption/ảnh) |
| Định dạng ảnh | JPEG |
| Nguồn captions | Amazon Mechanical Turk (crowd-sourced) |
| Ngôn ngữ | Tiếng Anh |

### 3.2 Phân chia tập dữ liệu

Dữ liệu được chia ngẫu nhiên với `random_seed = 42` để đảm bảo tái lập kết quả:

| Tập dữ liệu | Số ảnh | Tỷ lệ | Số caption |
|------------|--------|-------|-----------|
| Train | 29,769 | 93.7% | 148,844 |
| Validation | 1,014 | 3.2% | 5,070 |
| Test | 1,000 | 3.1% | 5,000 |
| **Tổng** | **31,783** | **100%** | **158,915** |

Các splits được lưu vào `workspace/flickr30k_splits.json` và **dùng chung cho cả ba mô hình**, đảm bảo so sánh công bằng.

### 3.3 Phân tích dữ liệu

**Phân tích ảnh:**
- Kích thước ảnh đa dạng (không đồng nhất), được resize về 224×224 trước khi đưa vào encoder
- Chủ đề ảnh đa dạng: người, động vật, phong cảnh, thể thao, hoạt động ngoài trời

**Phân tích caption:**
- Độ dài caption trung bình: ~12 từ/caption
- Sau khi thêm `startseq`/`endseq`: trung bình ~14 token
- Vocab thô (trước lọc): >15,000 từ
- Vocab được giới hạn ở **10,000 từ phổ biến nhất** để tránh sparse embedding

### 3.4 Tiền xử lý caption

Quy trình làm sạch caption (`clean_caption`):
1. Chuyển toàn bộ về chữ thường (lowercase)
2. Loại bỏ ký tự không phải chữ cái (giữ lại số nếu có nghĩa)
3. Giữ lại từ có độ dài ≥ 1 ký tự (bao gồm "a", "i")
4. Thêm token `startseq` ở đầu và `endseq` ở cuối
5. Lưu kết quả vào `workspace/descriptions_flickr30k.txt`

### 3.5 Tokenization và GloVe Embedding

- **Vocabulary**: 10,000 từ thường gặp nhất + 4 special tokens (`<pad>`, `<unk>`, `startseq`, `endseq`)
- **GloVe coverage**: 99.2% vocabulary được khởi tạo từ GloVe 300d (chỉ ~0.8% từ ngẫu nhiên)
- **Embedding matrix**: shape `(10,004, 300)`, fine-tunable trong quá trình training

---

## 4. KIẾN TRÚC MÔ HÌNH

### 4.1 Tổng quan kiến trúc chung

Cả ba mô hình đều theo kiến trúc **Encoder-Decoder với Bahdanau Attention**:

```
Ảnh đầu vào (224×224×3)
        │
        ▼
   [CNN/ViT Encoder]           (Frozen — không fine-tune)
        │
        ▼
 Features: (49, D)             D = 512 (VGG16, CLIP) hoặc 2048 (ResNet-101)
        │
        ├─────────────────────────────────────┐
        │                                     │
        ▼                                     ▼
 [GloVe Embedding (300d)]          [Bahdanau Attention]
        │                                     │
        ▼                                     │
   [LSTMCell (512)]  ←── context vector ──────┘
        │
        ▼
  [Linear Layer → Softmax]
        │
        ▼
  Phân phối xác suất trên Vocab (10,000 từ)
```

### 4.2 Chi tiết kiến trúc CaptionDecoder

**Lớp Embedding:**
- Input: token index `(batch, seq_len)`
- Output: `(batch, seq_len, 300)` — GloVe 300d, fine-tunable
- Dropout: 0.5

**Lớp LSTMCell:**
- Input: `[embedding (300d) + context (D/512d)]` → concat → Linear projection → 512d
- Hidden state: 512d
- Cell state: 512d

**Lớp BahdanauAttention:**
- `W_h`: Linear(512, 512) — chiếu hidden state
- `W_f`: Linear(D_feature, 512) — chiếu features ảnh
- `V`: Linear(512, 1) — tính attention score
- Output: context vector `(batch, D_feature)`

**Lớp Output:**
- Linear(512, 10,000) → Softmax

**Tổng số tham số**: ~11.9M parameters (VGG16 và CLIP) / ~14.5M (ResNet-101 do chiều features cao hơn)

### 4.3 Mô hình 1 — VGG16 + LSTM + Bahdanau Attention

| Thành phần | Chi tiết |
|-----------|---------|
| Encoder | VGG16 pre-trained ImageNet, trích xuất `conv5_3` |
| Feature shape | (49, 512) |
| Decoder | LSTMCell, hidden=512 |
| Attention | Bahdanau Additive |
| Embedding | GloVe 300d, fine-tunable |
| Frozen/Trainable | Encoder frozen, Decoder trainable |

### 4.4 Mô hình 2 — ResNet-101 + LSTM + Bahdanau Attention

| Thành phần | Chi tiết |
|-----------|---------|
| Encoder | ResNet-101 pre-trained ImageNet, trích xuất `layer4` |
| Feature shape | (49, 2048) |
| Decoder | LSTMCell, hidden=512 |
| Attention | Bahdanau Additive |
| Embedding | GloVe 300d, fine-tunable |
| Frozen/Trainable | Encoder frozen, Decoder trainable |

ResNet-101 cung cấp đặc trưng **2048 chiều** thay vì 512d của VGG16, phong phú và biểu đạt hơn nhờ kiến trúc sâu 101 lớp với residual connections.

### 4.5 Mô hình 3 — CLIP ViT-B/32 + LSTM + Bahdanau Attention

| Thành phần | Chi tiết |
|-----------|---------|
| Encoder | CLIP ViT-B/32, pre-trained trên 400M cặp (image, text) |
| Feature shape | (49, 512) — 49 patch tokens |
| Decoder | LSTMCell, hidden=512 |
| Attention | Bahdanau Additive |
| Embedding | GloVe 300d, fine-tunable |
| Frozen/Trainable | CLIP frozen, Decoder trainable |

CLIP khác biệt cơ bản so với VGG16/ResNet-101: features của CLIP không chỉ mã hóa thông tin thị giác mà còn mang **ngữ nghĩa ngôn ngữ-thị giác** (vision-language semantics) nhờ pre-training contrastive. CLIP chia ảnh thành các patch 32×32, mỗi patch trở thành một token, sau đó xử lý qua cơ chế self-attention của Transformer.

---

## 5. CHI TIẾT THỰC NGHIỆM

### 5.1 Môi trường thực nghiệm

| Thông số | Giá trị |
|---------|--------|
| Phần cứng | MacBook Apple Silicon (MPS — Metal Performance Shaders) |
| Framework | PyTorch |
| Python | 3.10 (Conda env `tf-metal`) |
| Accelerator | Apple Neural Engine / MPS GPU |

### 5.2 Trích xuất đặc trưng ảnh (Feature Extraction)

Để tiết kiệm thời gian, đặc trưng ảnh được trích xuất **một lần duy nhất** và lưu vào file pickle:

| Mô hình | File lưu trữ | Tốc độ trích xuất |
|---------|-------------|-----------------|
| VGG16 | `features_conv_flickr30k.pkl` | ~15-20 phút |
| ResNet-101 | `features_resnet101_flickr30k.pkl` | ~18-22 phút |
| CLIP | `features_clip_flickr30k.pkl` | ~4 phút 21 giây (3.81 img/s) |

CLIP nhanh nhất nhờ ViT-B/32 được tối ưu tốt.

### 5.3 Hyperparameters

| Hyperparameter | VGG16 | ResNet-101 | CLIP |
|---------------|-------|-----------|------|
| Vocab size | 10,000 | 10,000 | 10,000 |
| Max sequence length | 35 | 25 | 35 |
| Embedding dim | 300 | 300 | 300 |
| LSTM hidden size | 512 | 512 | 512 |
| Batch size | 64 | 32 | 64 |
| Epochs | 50 | 10 | 10 |
| Learning rate | 1e-4 | 1e-4 | 1e-4 |
| Weight decay | 1e-4 | 1e-4 | 1e-4 |
| Label smoothing | 0.1 | 0.1 | 0.1 |
| Gradient clipping | — | — | 5.0 |
| Beam width | 5 | 5 | 5 |

**Lưu ý:** ResNet-101 giảm `max_length` từ 35 xuống 25 (giảm ~28% thời gian/epoch) và `batch_size` từ 64 xuống 32 do bộ nhớ features 2048d lớn hơn. CLIP thêm gradient clipping (5.0) để ổn định training.

### 5.4 Hàm mất mát và Tối ưu hoá

**Loss function**: `CrossEntropyLoss` với:
- `ignore_index = 0` (bỏ qua padding token)
- `label_smoothing = 0.1` (giảm overconfidence, cải thiện generalization)

**Optimizer**: AdamW với:
- `lr = 1e-4`, `weight_decay = 1e-4`

**Learning Rate Scheduler**: ReduceLROnPlateau:
- `mode = 'min'`, monitor `val_loss`
- `factor = 0.5` (giảm LR 50% khi val_loss không cải thiện)
- `patience = 2` (chờ 2 epoch trước khi giảm LR)

**Early Stopping**:
- VGG16: `patience = 7` (tối đa 50 epochs)
- ResNet-101: `patience = 3` (tối đa 10 epochs)
- CLIP: `patience = 5` (tối đa 10 epochs)

### 5.5 Quá trình huấn luyện

#### VGG16 Training History (các epoch quan trọng)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Status |
|-------|-----------|----------|----------|---------|--------|
| 1 | 5.3729 | 25.2% | 4.7449 | 31.1% | — |
| 2 | 4.7301 | 31.0% | 4.4441 | 33.7% | — |
| 3 | 4.5138 | 32.8% | 4.2903 | 35.0% | — |
| 10 | 4.0369 | 37.9% | 3.9641 | 38.7% | — |
| 20 | 3.7985 | 41.3% | 3.8890 | 39.5% | — |
| **23** | **3.7505** | **42.1%** | **3.8851** | **39.6%** | **BEST** |
| 30 | 3.6519 | 43.7% | 3.8933 | 39.5% | Early stop |

Training dừng tại epoch 30 (7 epoch sau best tại epoch 23). Best checkpoint được lưu tại `val_loss = 3.8851`.

#### ResNet-101 Training History (toàn bộ)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Thời gian |
|-------|-----------|----------|----------|---------|----------|
| 1 | 4.6620 | 31.9% | 4.3112 | 35.0% | 1203s |
| 2 | 4.3482 | 34.7% | 4.1331 | 36.8% | 1201s |
| 3 | 4.1834 | 36.5% | 4.0392 | 37.8% | 1205s |
| 4 | 4.0723 | 37.9% | 3.9819 | 38.5% | 1212s |
| 5 | 3.9874 | 39.0% | 3.9436 | 39.1% | 1194s |
| 6 | 3.9162 | 40.0% | 3.9197 | 39.3% | 1198s |
| 7 | 3.8573 | 40.9% | 3.9040 | 39.5% | 1189s |
| 8 | 3.8045 | 41.7% | 3.8914 | 39.7% | 1182s |
| 9 | 3.7589 | 42.4% | 3.8875 | 39.6% | 1176s |
| **10** | **3.7171** | **43.0%** | **3.8775** | **39.8%** | 1180s **BEST** |

Val_loss giảm đều qua toàn bộ 10 epochs, chứng tỏ mô hình vẫn còn tiếp tục học được. Best checkpoint: `val_loss = 3.8775`.

#### CLIP Training History (tóm tắt)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Thời gian |
|-------|-----------|----------|----------|---------|----------|
| 1 | 5.4206 | 24.8% | 4.8027 | 30.3% | 609s |
| 2 | 4.7651 | 30.9% | 4.4603 | 33.8% | 602s |
| 3 | 4.5195 | 33.2% | 4.2786 | 35.4% | 603s |
| ... | ... | ... | ... | ... | ~605s/epoch |
| **10** | — | — | **3.8817** | — | — **BEST** |

CLIP nhanh nhất trong ba mô hình (~609s/epoch so với ~1190s của ResNet-101), nhờ features chỉ 512d và batch_size 64. Best checkpoint: `val_loss = 3.8817`.

---

## 6. KẾT QUẢ VÀ ĐÁNH GIÁ

### 6.1 BLEU Scores trên tập Test

Sau khi huấn luyện, mô hình được đánh giá trên tập test (1,000 ảnh × 5 caption tham chiếu) bằng Beam Search (beam_width = 5):

| Mô hình | BLEU-1 | BLEU-2 | BLEU-3 | **BLEU-4** |
|---------|--------|--------|--------|------------|
| Flickr8k Baseline (VGG16 cũ) | 0.5285 | 0.3385 | 0.2097 | 0.1186 |
| **VGG16 + Attention** | 0.6159 | 0.4364 | 0.3049 | **0.2125** |
| **ResNet-101 + Attention** | 0.6808 | 0.4948 | 0.3502 | **0.2484** |
| **CLIP ViT-B/32 + Attention** | 0.6598 | 0.4830 | 0.3443 | **0.2433** |

### 6.2 Cải thiện so với Baseline

| Mô hình | BLEU-4 | Cải thiện so với Flickr8k Baseline | So với VGG16 |
|---------|--------|-----------------------------------|-------------|
| VGG16 | 0.2125 | +79.2% (+0.0939) | — |
| ResNet-101 | 0.2484 | +109.4% (+0.1298) | +16.9% |
| CLIP | 0.2433 | +105.1% (+0.1247) | +14.5% |

Tất cả ba mô hình đều vượt xa baseline Flickr8k (0.1186), với:
- VGG16 cải thiện **+79.2%**
- ResNet-101 cải thiện **+109.4%**
- CLIP cải thiện **+105.1%**

### 6.3 Val Loss so sánh

| Mô hình | Best Val Loss | Epoch đạt được |
|---------|-------------|--------------|
| VGG16 | 3.8851 | Epoch 23/50 |
| ResNet-101 | 3.8775 | Epoch 10/10 (chưa hội tụ) |
| CLIP | 3.8817 | Epoch 10/10 (chưa hội tụ) |

ResNet-101 đạt val_loss thấp nhất (3.8775), tương ứng với BLEU-4 cao nhất.

### 6.4 Ví dụ caption được sinh ra (VGG16)

| Ảnh ID | Caption sinh ra | BLEU-4 |
|--------|----------------|--------|
| 2374289148 | "a man and a woman are walking down the street" | 0.200 |
| 3174417550 | "a man in a black shirt is jumping into a water fountain" | 0.235 |
| 2178295140 | "a woman and a child are standing in front of a store" | 0.257 |
| 3847158742 | "a person in a red shirt is walking down a rocky path" | 0.279 |
| 6758527995 | "a man and a woman are dancing on a dance floor" | 0.603 |

### 6.5 So sánh tốc độ và tài nguyên

| Mô hình | Thời gian/epoch | Tổng training | RAM features | BLEU-4 |
|---------|----------------|--------------|-------------|--------|
| VGG16 | ~620s | ~18,600s (23 epochs) | ~300MB | 0.2125 |
| ResNet-101 | ~1,190s | ~11,900s (10 epochs) | ~1.2GB | 0.2484 |
| CLIP | ~606s | ~6,060s (10 epochs) | ~300MB | 0.2433 |

**CLIP có hiệu quả cao nhất**: tốc độ training nhanh nhất trong ba mô hình, bộ nhớ tương đương VGG16, nhưng BLEU-4 gần ngang ResNet-101.

---

## 7. PHÂN TÍCH VÀ THẢO LUẬN

### 7.1 Tại sao ResNet-101 đạt BLEU-4 cao nhất?

ResNet-101 vượt VGG16 (+16.9% BLEU-4) vì:
1. **Đặc trưng phong phú hơn**: 2048 chiều vs 512 chiều — biểu diễn ảnh phong phú và biểu đạt hơn
2. **Kiến trúc sâu hơn**: 101 lớp vs 16 lớp, học được đặc trưng phân cấp phức tạp hơn
3. **Residual connections**: Giải quyết gradient vanishing, dẫn đến đặc trưng trừu tượng hơn

ResNet-101 vượt CLIP (+2.1% BLEU-4) dù features cùng số vị trí không gian (49), vì:
- 2048d cho phép attention head phân biệt các vùng ảnh tinh tế hơn
- CLIP tối ưu cho image-text retrieval, không phải generation

### 7.2 Tại sao CLIP cạnh tranh được dù features 512d?

CLIP (0.2433 BLEU-4) gần ngang ResNet-101 (0.2484) dù features chỉ 512d vì:
1. **Vision-Language alignment**: Mỗi patch token của CLIP đã mang thông tin ngữ nghĩa liên kết với ngôn ngữ, phù hợp hơn cho text generation
2. **Pre-training scale**: 400M cặp (ảnh, text) — học được biểu diễn ảnh phong phú về mặt ngữ nghĩa
3. **ViT architecture**: Self-attention trong Vision Transformer cho phép học các mối quan hệ toàn cục giữa các patch, khác CNN chỉ học cục bộ

### 7.3 Phân tích lỗi phổ biến

Dựa trên phân tích error trong notebook demo:

**Lỗi thường gặp:**
1. **Confusion về màu sắc/số lượng**: "a man" thay vì "two men", "red shirt" thay vì "blue shirt"
2. **Thiếu chi tiết phụ**: Bỏ sót background objects hoặc context
3. **Caption quá ngắn hoặc quá generic**: "a dog is running" khi thực ra có nhiều chi tiết hơn
4. **Repetition**: Đôi khi lặp từ nếu beam search không được điều chỉnh tốt

**Caption tốt:** Các ảnh có chủ thể rõ ràng, màu sắc tương phản, hoạt động đặc trưng

### 7.4 Ảnh hưởng của Label Smoothing (ε = 0.1)

Label smoothing chuyển phân phối target từ one-hot sang:
- `p(correct) = 1 - ε = 0.9`
- `p(other) = ε / (V-1) ≈ 0.00001`

Điều này giúp mô hình không quá confident, cải thiện generalization trên tập test và giảm overfitting.

### 7.5 Ảnh hưởng của Beam Search (width = 5)

Beam search (width=5) so với greedy search cải thiện BLEU-4 đáng kể:
- Greedy: chọn từ max-prob tại mỗi bước → local optimum
- Beam search: duy trì 5 ứng viên → tìm được chuỗi có global probability cao hơn

### 7.6 Hạn chế của mô hình hiện tại

1. **Max length cố định**: ResNet-101 dùng max_length=25, có thể cắt bớt caption dài
2. **Frozen encoder**: Không fine-tune encoder với task-specific signal
3. **Chỉ dùng static GloVe**: Không dùng contextual embeddings (BERT, RoBERTa)
4. **Dataset đơn ngữ**: Chỉ tiếng Anh
5. **Beam search đơn giản**: Chưa có length penalty, repetition penalty

---

## 8. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 8.1 Kết luận

Đề tài đã nghiên cứu và thực nghiệm thành công bài toán **sinh mô tả ảnh tự động** trên dataset Flickr30k với ba hướng tiếp cận khác nhau, đều dựa trên kiến trúc **Encoder-Decoder + Bahdanau Attention**:

| Mô hình | BLEU-4 | Nhận xét |
|---------|--------|---------|
| VGG16 + LSTM + Attention | 0.2125 | Baseline mạnh, gấp 1.79× Flickr8k cũ |
| ResNet-101 + LSTM + Attention | **0.2484** | **Tốt nhất**, features 2048d phong phú |
| CLIP ViT-B/32 + LSTM + Attention | 0.2433 | Cạnh tranh tốt, training nhanh nhất |

**Kết quả nổi bật:**
- Tất cả ba mô hình đều vượt xa Flickr8k baseline (0.1186) từ 79% đến 109%
- ResNet-101 đạt **BLEU-4 = 0.2484** — cao nhất trong ba mô hình
- CLIP cho thấy tiềm năng của vision-language pre-training trong image captioning
- Kiến trúc Bahdanau Attention hiệu quả trong việc tập trung vào vùng ảnh liên quan

### 8.2 Hướng phát triển

1. **Fine-tune encoder**: Thay vì frozen, cho phép encoder cập nhật weights với task-specific signal
2. **Transformer Decoder**: Thay LSTM bằng Transformer decoder (như GPT-2) để xử lý long-range dependency tốt hơn
3. **CLIP end-to-end fine-tuning**: Fine-tune cả CLIP encoder với captioning loss
4. **Larger vocab / Subword tokenization**: Dùng BPE hoặc SentencePiece thay vì word-level tokenization
5. **Data Augmentation**: Augment ảnh (flip, crop, color jitter) để tăng robustness
6. **Cross-attention Transformer**: Kiến trúc hiện đại hơn (BEiT, BLIP, OFA) đạt SOTA trên các benchmark
7. **Đa ngôn ngữ**: Mở rộng sang tiếng Việt với dataset tự xây dựng

---

## 9. TÀI LIỆU THAM KHẢO

[1] Vinyals, O., Toshev, A., Bengio, S., & Erhan, D. (2015). **Show and tell: A neural image caption generator**. CVPR 2015.

[2] Xu, K., Ba, J., Kiros, R., et al. (2015). **Show, attend and tell: Neural image caption generation with visual attention**. ICML 2015.

[3] Bahdanau, D., Cho, K., & Bengio, Y. (2015). **Neural machine translation by jointly learning to align and translate**. ICLR 2015.

[4] Simonyan, K., & Zisserman, A. (2014). **Very deep convolutional networks for large-scale image recognition** (VGG16). ICLR 2015.

[5] He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep residual learning for image recognition** (ResNet). CVPR 2016.

[6] Radford, A., Kim, J. W., Hallacy, C., et al. (2021). **Learning transferable visual models from natural language supervision** (CLIP). ICML 2021.

[7] Pennington, J., Socher, R., & Manning, C. D. (2014). **GloVe: Global vectors for word representation**. EMNLP 2014.

[8] Young, P., Lai, A., Hodosh, M., & Hockenmaier, J. (2014). **From image descriptions to visual denotations** (Flickr30k). TACL 2014.

[9] Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). **BLEU: A method for automatic evaluation of machine translation**. ACL 2002.

[10] Loshchilov, I., & Hutter, F. (2019). **Decoupled weight decay regularization** (AdamW). ICLR 2019.

---

*Báo cáo được viết dựa trên kết quả thực nghiệm từ ba notebook:*
- *`ImageCaptioning_Flickr30k_VGG16.ipynb`*
- *`ImageCaptioning_Flickr30k_ResNet101.ipynb`*
- *`ImageCaptioning_Flickr30k_CLIP.ipynb`*
