# SLIDE BẢO VỆ ĐỒ ÁN — NEURAL IMAGE CAPTION GENERATION

> **Cách dùng file này:** Mở VSCode → nhấn `Cmd+Shift+V` để xem Markdown Preview → Full-screen → trình chiếu.
> Mỗi phần `---` là một slide. Phần `> 🗣️ NÓI:` là gợi ý lời thoại, **không hiển thị khi trình chiếu** (ẩn trong preview).
> **Cảnh** trình bày Slide 1–11 | **Giang** trình bày Slide 12–20

---

---

# SLIDE 1 — TRANG BÌA

<br>

# SINH MÔ TẢ ẢNH TỰ ĐỘNG

## SỬ DỤNG HỌC SÂU

### Neural Image Caption Generation

<br>

|                          |                                                        |
| ------------------------ | ------------------------------------------------------ |
| **Sinh viên thực hiện**  | Lê Văn Cảnh &nbsp;&nbsp;&nbsp; Nguyễn Đức Trường Giang |
| **Lớp**                  | DT2307L                                                |
| **Giảng viên hướng dẫn** | Hồ Nhựt Minh                                           |
| **Trường**               | Aptech                                                 |
| **Năm học**              | 2025 – 2026                                            |

---

> 🗣️ **:** "Thưa Thầy.
> Nhóm chúng em gồm Lê Văn Cảnh và Nguyễn Đức Trường Giang, lớp DT2307L.
> Hôm nay nhóm em xin trình bày đồ án: **Sinh mô tả ảnh tự động sử dụng Học sâu**."

---

---

# SLIDE 2 — NỘI DUNG TRÌNH BÀY

<br>

| #   | Nội dung                          |
| --- | --------------------------------- |
| 1   | Đặt vấn đề — Tại sao làm bài này? |
| 2   | Dataset — Flickr30k               |
| 3   | Công nghệ sử dụng — Tại sao chọn? |
| 4   | Kiến trúc mô hình                 |
| 5   | Quá trình huấn luyện              |
| 6   | Kết quả & Demo                    |
| 7   | Phân tích — Kết luận              |

---

> 🗣️ **:** "Bài trình bày của nhóm em gồm 7 phần chính như trên màn hình.
> Em sẽ trình bày từ Phần 1 đến Phần 4, sau đó bạn Giang sẽ tiếp tục từ Phần 5."

---

---

# SLIDE 3 — ĐẶT VẤN ĐỀ

## Bài toán: Máy tính có thể "đọc" và mô tả ảnh không?

<br>

```
🖼️  [Ảnh một người đang chạy]
              ↓
🤖  "A man is running on a track"
```

<br>

### Ứng dụng thực tế:

| Ứng dụng                  | Mô tả                                |
| ------------------------- | ------------------------------------ |
| 👁️ Hỗ trợ người khiếm thị | Đọc nội dung ảnh qua giọng nói       |
| 🔍 Tìm kiếm ảnh           | Tìm ảnh bằng mô tả ngôn ngữ tự nhiên |
| 📱 Mạng xã hội            | Tự động gợi ý caption cho ảnh        |
| 🤖 Robot thông minh       | Robot hiểu và mô tả môi trường       |
| 🏥 Y tế                   | Tự động mô tả ảnh X-quang, MRI       |

---

> 🗣️ **:** "Thưa Thầy, bài toán chúng em thực hiện là **Sinh mô tả ảnh tự động** —
> tức là đưa vào một bức ảnh, hệ thống sẽ **tự động sinh ra một câu mô tả bằng ngôn ngữ tự nhiên** mà không cần con người can thiệp.
> Bài toán này giao thoa giữa hai lĩnh vực: **Thị giác máy tính** và **Xử lý ngôn ngữ tự nhiên**.
> Ứng dụng rất rộng như hỗ trợ người khiếm thị, tìm kiếm ảnh, y tế..."

---

---

# SLIDE 4 — MỤC TIÊU ĐỀ TÀI

## Nhóm em nghiên cứu và so sánh 3 hướng tiếp cận:

<br>

| Mô hình       | Encoder       | Decoder          | Đặc điểm                       |
| ------------- | ------------- | ---------------- | ------------------------------ |
| **Mô hình 1** | VGG16         | LSTM + Attention | Baseline — phổ biến, đơn giản  |
| **Mô hình 2** | ResNet-101    | LSTM + Attention | Đặc trưng sâu hơn (2048 chiều) |
| **Mô hình 3** | CLIP ViT-B/32 | LSTM + Attention | Ngữ nghĩa vision-language      |

<br>

> ✅ Cả 3 mô hình: **cùng dataset, cùng decoder, cùng tập test** → So sánh **công bằng**

---

> 🗣️ **:** "Mục tiêu của nhóm em là **xây dựng và so sánh 3 mô hình** sinh mô tả ảnh.
> Ba mô hình đều dùng chung kiến trúc decoder LSTM + Bahdanau Attention, chỉ khác nhau ở **phần Encoder** — tức phần trích xuất đặc trưng từ ảnh.
> Điều này giúp so sánh **công bằng** xem Encoder nào cho kết quả tốt hơn."

---

---

# SLIDE 5 — DATASET: FLICKR30K

## Dữ liệu huấn luyện

> Young et al., 2014 — Một trong những dataset chuẩn nhất cho bài toán Image Captioning

<br>

| Thông số    | Giá trị                                      |
| ----------- | -------------------------------------------- |
| Tổng số ảnh | **31,783 ảnh**                               |
| Số mô tả    | **158,915 câu** (5 câu/ảnh)                  |
| Định dạng   | JPEG — đa dạng chủ đề                        |
| Nguồn nhãn  | Amazon Mechanical Turk (người thật gán nhãn) |
| Ngôn ngữ    | Tiếng Anh                                    |

<br>

### Phân chia dữ liệu (seed = 42 — đảm bảo tái lập kết quả):

| Tập            | Số ảnh | Tỷ lệ | Số caption |
| -------------- | ------ | ----- | ---------- |
| **Train**      | 29,769 | 93.7% | 148,844    |
| **Validation** | 1,014  | 3.2%  | 5,070      |
| **Test**       | 1,000  | 3.1%  | 5,000      |

📌 **3 mô hình dùng chung tập Test → So sánh công bằng**

---

> 🗣️ **:** "Dataset nhóm em dùng là **Flickr30k** — được công bố bởi Young và cộng sự năm 2014,
> gồm gần 32 nghìn ảnh, mỗi ảnh có **5 câu mô tả khác nhau** do người thật viết.
> Dữ liệu được chia ngẫu nhiên với seed cố định 42 để đảm bảo mọi thực nghiệm đều có thể tái lập.
> Quan trọng là cả 3 mô hình đều dùng **chính xác cùng tập test 1,000 ảnh** — nên kết quả so sánh là công bằng."

---

---

# SLIDE 6 — TIỀN XỬ LÝ DỮ LIỆU

## Pipeline xử lý caption

```
Raw caption: "A man Running in the Park!!"
       ↓  (lowercase)
       "a man running in the park"
       ↓  (loại ký tự đặc biệt)
       "a man running in the park"
       ↓  (thêm token bắt đầu/kết thúc)
       "startseq a man running in the park endseq"
```

<br>

### Từ điển (Vocabulary):

| Thông số                | Giá trị                          |
| ----------------------- | -------------------------------- |
| Vocab thô (trước lọc)   | > 23,457 từ                      |
| Vocab sau lọc           | **10,000 từ** phổ biến nhất      |
| Giảm                    | 57.4% — loại bỏ từ hiếm gặp      |
| **GloVe 300d coverage** | **99.2%** được khởi tạo từ GloVe |

<br>

> 📖 **GloVe** (Pennington et al., 2014) — Vector từ 300 chiều, huấn luyện trên **6 tỷ token** từ Wikipedia

---

> 🗣️ **:** "Trước khi huấn luyện, caption phải được làm sạch theo 3 bước chính.
> Kết quả là từ hơn 23,000 từ trong kho dữ liệu, chúng em giữ lại 10,000 từ thường gặp nhất.
> Mỗi từ trong từ điển được **khởi tạo bằng GloVe 300d** — một bộ vector từ được huấn luyện sẵn trên 6 tỷ từ.
> Điều này giúp mô hình **không phải học từ đầu về ngữ nghĩa của từ**, 99.2% vocab có sẵn vector GloVe."

---

---

# SLIDE 7 — KIẾN TRÚC TỔNG QUAN

## Encoder – Decoder với Bahdanau Attention

```
                    ┌─────────────┐
  Ảnh 224×224×3  →  │   ENCODER   │  →  Features (49, D)
                    │ CNN / ViT   │
                    └─────────────┘
                           │
                           ▼
                    ┌─────────────────────────────────┐
                    │           DECODER               │
  Caption (từng     │  GloVe Embedding (300d)         │
  từ một)       →  │         ↓                       │
                    │    LSTMCell (512)  ←── context  │
                    │         ↓              ↑        │
                    │    Output Projection   │        │
                    │         ↓             │        │
                    │   Softmax(10,000)  [Attention]  │
                    └─────────────────────────────────┘
```

- **D = 512** cho VGG16 và CLIP
- **D = 2048** cho ResNet-101

---

> 🗣️ **:** "Đây là kiến trúc tổng quan của cả 3 mô hình.
> Phần **Encoder** nhận ảnh vào và xuất ra một ma trận đặc trưng kích thước 49 × D — tức 49 vị trí không gian trong ảnh.
> Phần **Decoder** là LSTM nhận từng từ, kết hợp với **Attention** để biết nên nhìn vào vùng nào của ảnh khi sinh từ tiếp theo.
> Ba mô hình giống nhau hoàn toàn ở Decoder, chỉ khác ở Encoder."

---

---

# SLIDE 8 — ENCODER 1: VGG16

## Tại sao dùng VGG16?

> Simonyan & Zisserman, 2014 — Giải nhất ImageNet 2014

<br>

| Đặc điểm             | Giá trị                                 |
| -------------------- | --------------------------------------- |
| Kiến trúc            | 16 lớp CNN, đơn giản, đồng nhất         |
| Lớp trích xuất       | `conv5_3` (block cuối cùng)             |
| Feature shape        | **(49, 512)** — 49 vị trí × 512 kênh    |
| Pre-trained          | ImageNet — 1.2 triệu ảnh, 1000 lớp      |
| Vai trò trong đề tài | **Baseline** — mô hình cơ sở để so sánh |

<br>

```
Ảnh 224×224  →  [Conv Block 1-5]  →  Feature Map 7×7×512  →  reshape  →  (49, 512)
```

> ✅ Đơn giản, được kiểm chứng rộng rãi, phù hợp làm baseline

---

> 🗣️ **:** "Mô hình đầu tiên dùng **VGG16** làm Encoder.
> VGG16 là mạng CNN 16 lớp, giải nhất ImageNet 2014, rất nổi tiếng và được dùng rộng rãi.
> Sau khi qua 5 khối Conv, ảnh 224×224 tạo ra feature map 7×7×512, chúng em reshape thành 49 vector 512 chiều.
> Nhóm em chọn VGG16 làm **baseline** vì nó đơn giản, kết quả đáng tin cậy, dễ so sánh với các nghiên cứu khác."

---

---

# SLIDE 9 — ENCODER 2: RESNET-101

## Tại sao nâng cấp lên ResNet-101?

> He et al., 2016 — Giải nhất ImageNet 2015, CVPR Best Paper

<br>

| Đặc điểm       | Giá trị                                                   |
| -------------- | --------------------------------------------------------- |
| Kiến trúc      | 101 lớp — **Residual Connections** (kết nối tắt)          |
| Lớp trích xuất | `layer4`                                                  |
| Feature shape  | **(49, 2048)** — **gấp 4 lần VGG16**                      |
| Ưu điểm chính  | Học được đặc trưng sâu hơn mà không bị vanishing gradient |

<br>

```
VGG16:     (49, 512)  ←  ít thông tin hơn
ResNet-101: (49, 2048) ←  đặc trưng phong phú gấp 4 lần
```

**Residual connection:** Nếu lớp hiện tại không học được gì → truyền thẳng kết quả lớp trước qua (skip connection) → không bị mất thông tin

---

> 🗣️ **:** "Mô hình thứ hai nâng cấp lên **ResNet-101** với 101 lớp.
> Điểm đặc biệt là **residual connections** — kết nối tắt giúp mạng rất sâu vẫn học được tốt mà không bị mất gradient.
> Quan trọng hơn, feature output là **(49, 2048)** — nhiều thông tin gấp 4 lần VGG16.
> Câu hỏi đặt ra là: đặc trưng phong phú hơn có cho caption tốt hơn không? Kết quả sẽ nói lên điều này."

---

---

# SLIDE 10 — ENCODER 3: CLIP ViT-B/32

## Tại sao thử CLIP?

> Radford et al., 2021 — OpenAI — Contrastive Language-Image Pretraining

<br>

| Đặc điểm         | Giá trị                                                                       |
| ---------------- | ----------------------------------------------------------------------------- |
| Kiến trúc        | Vision Transformer (ViT-B/32)                                                 |
| Huấn luyện trên  | **400 triệu cặp (ảnh, văn bản)** từ internet                                  |
| Phương pháp      | Contrastive Learning — ảnh và caption phải "gần nhau" trong không gian vector |
| Feature shape    | **(49, 512)**                                                                 |
| Đặc điểm nổi bật | Feature đã chứa **ngữ nghĩa ngôn ngữ** — không chỉ visual                     |

<br>

```
VGG16 / ResNet: Chỉ học từ ảnh (ImageNet — phân loại 1000 lớp)
CLIP:           Học đồng thời từ ÂNH + VĂN BẢN (400M cặp) → hiểu ngữ nghĩa
```

> ✅ Lý thuyết: CLIP nên tốt hơn vì feature đã align với ngôn ngữ

---

> 🗣️ **:** "Mô hình thứ ba dùng **CLIP** của OpenAI — đây là điểm mới và thú vị nhất của đề tài.
> CLIP được huấn luyện trên **400 triệu cặp ảnh-văn bản** từ internet, không phải ImageNet.
> Kết quả là feature của CLIP không chỉ chứa thông tin thị giác mà còn **chứa ngữ nghĩa ngôn ngữ**.
> Về lý thuyết, CLIP nên tạo ra caption tốt hơn vì feature đã 'quen' với ngôn ngữ rồi.
> Kết quả thực tế — các bạn sẽ thấy ngay sau đây — có điều bất ngờ."

---

---

# SLIDE 11 — CƠ CHẾ BAHDANAU ATTENTION

## Tại sao cần Attention?

```
Không có Attention:          Có Attention:
"A man is running"           "A man is running"
   ↑                            ↑
Chỉ nhìn 1 vector tổng      Nhìn đúng vùng "người đang chạy"
của toàn bộ ảnh              khi sinh từ "man" và "running"
```

<br>

### Công thức Bahdanau Attention:

> _Bahdanau et al., 2014 — "Neural Machine Translation by Jointly Learning to Align and Translate"_

```
e_i  =  V^T · tanh(W_h · h_{t-1}  +  W_f · f_i)    ← điểm số vùng ảnh thứ i
α_i  =  softmax(e_i)                                  ← trọng số chú ý (tổng = 1)
c_t  =  Σ α_i · f_i                                   ← context vector
```

- `h_{t-1}` : trạng thái ẩn LSTM bước trước
- `f_i` : đặc trưng ảnh tại vị trí không gian thứ i (i = 1..49)
- `α_i` : **mức độ chú ý** tại vị trí i — càng cao càng quan trọng
- `c_t` : **context vector** — tóm tắt thông tin ảnh cần thiết tại bước t

📌 **Hiển thị heatmap attention từ notebook**

---

> 🗣️ **:** "Bahdanau Attention là cơ chế cho phép mô hình **nhìn vào đúng vùng ảnh** khi sinh mỗi từ.
> Ví dụ khi sinh từ 'dog', mô hình sẽ tập trung vào vùng có con chó trong ảnh.
> Công thức gồm 3 bước: tính điểm số cho 49 vùng ảnh, chuẩn hóa thành xác suất bằng softmax, rồi lấy trung bình có trọng số.
> Đây là công bố năm 2014 của Bahdanau — được trích dẫn hơn 20,000 lần — rất uy tín trong cộng đồng NLP."

---

---

# SLIDE 12 — HUẤN LUYỆN: TEACHER FORCING & BEAM SEARCH

## Teacher Forcing — Khi huấn luyện

```
Ground truth:  startseq  a  man  is  running  endseq
Đầu vào LSTM:  startseq  a  man  is  running    ↑
Đầu ra LSTM:      a     man  is  running  endseq
```

> ✅ Dù LSTM đoán sai ở bước trước → vẫn nhận **từ đúng** làm input → Huấn luyện ổn định, hội tụ nhanh

<br>

## Beam Search — Khi sinh caption (Inference)

```
Greedy: chọn từ có xác suất cao nhất tại mỗi bước → dễ bị cục bộ tối ưu

Beam Search (width = 5):
  Bước 1: Giữ 5 ứng viên tốt nhất: "a", "the", "man", "two", "group"
  Bước 2: Mở rộng 5×vocab → Giữ 5 tốt nhất tiếp theo
  ...
  Chọn chuỗi có log-probability tổng cao nhất
```

> ✅ Kết quả tốt hơn greedy vì tìm kiếm **rộng hơn** trong không gian caption

---

> 🗣️ **:** "Phần tiếp theo từ quá trình huấn luyện.
> Trong lúc huấn luyện, nhóm em dùng **Teacher Forcing** — tức là dù mô hình đoán sai, bước tiếp theo vẫn nhận từ đúng làm input.
> Điều này giúp quá trình học ổn định hơn nhiều so với dùng chính dự đoán của mô hình.
> Khi inference, nhóm em dùng **Beam Search** với beam width 5 — giữ 5 ứng viên tốt nhất tại mỗi bước thay vì chỉ chọn 1 từ."

---

---

# SLIDE 13 — QUÁ TRÌNH HUẤN LUYỆN

## Hyperparameters (giống nhau cho cả 3 mô hình)

| Tham số          | Giá trị                                              |
| ---------------- | ---------------------------------------------------- |
| Vocab size       | 10,000                                               |
| Embedding dim    | 300 (GloVe 300d)                                     |
| LSTM hidden size | 512                                                  |
| Dropout          | 0.5                                                  |
| Batch size       | 64                                                   |
| Learning rate    | 1e-4 (AdamW)                                         |
| LR Scheduler     | ReduceLROnPlateau (×0.5 nếu không cải thiện 3 epoch) |
| Early stopping   | Patience = 7 epoch                                   |
| Beam width       | 5                                                    |

<br>

### Số epoch thực tế:

| Mô hình    | Epochs        | Thời gian / epoch | Ghi chú                       |
| ---------- | ------------- | ----------------- | ----------------------------- |
| VGG16      | **30 epochs** | ~400 giây         | Early stopping epoch 30       |
| ResNet-101 | **10 epochs** | ~1,200 giây       | Hội tụ nhanh hơn              |
| CLIP       | **10 epochs** | ~900 giây         | Feature extraction nhanh nhất |

📌 **Hiển thị Training Loss Curves từ notebook (3 mô hình)**

---

> 🗣️ **:** "Cả 3 mô hình dùng chung bộ hyperparameters để đảm bảo so sánh công bằng.
> Điểm thú vị là VGG16 cần **30 epochs** để hội tụ, trong khi ResNet-101 và CLIP chỉ cần **10 epochs**.
> Điều này cho thấy đặc trưng của ResNet và CLIP **phong phú hơn**, mô hình học nhanh hơn.
> Nhóm em dùng **AdamW** thay vì Adam thông thường vì có thêm weight decay, giảm overfitting tốt hơn."

---

---

# SLIDE 14 — THƯỚC ĐO BLEU SCORE

## BLEU là gì và tại sao dùng?

> Papineni et al., 2002 — IBM — Được dùng chuẩn trong hơn 20 năm cho NLP

<br>

### Ý nghĩa đơn giản:

```
Caption sinh ra:  "a dog is running in the park"
Ground truth:     "a brown dog runs across the grass"

BLEU-1 (1-gram): "a", "dog", "in", "the" → khớp 4/7 → 57%
BLEU-2 (2-gram): "a dog" → khớp 1/6 → 17%
BLEU-4 (4-gram): 4 từ liên tiếp phải khớp → Khó nhất, chuẩn nhất
```

<br>

| Thước đo | Ý nghĩa                      | Độ khó                 |
| -------- | ---------------------------- | ---------------------- |
| BLEU-1   | Từ đơn khớp với ground truth | Dễ                     |
| BLEU-2   | Cặp từ khớp                  | Trung bình             |
| BLEU-4   | 4 từ liên tiếp khớp          | Khó — **chỉ số chính** |

> 📌 Đánh giá trên **1,000 ảnh × 5 caption/ảnh** = 5,000 câu tham chiếu → Kết quả đáng tin cậy

---

> 🗣️ **:** "Để đánh giá chất lượng caption, nhóm em dùng **BLEU score** — thước đo chuẩn của NLP từ năm 2002.
> BLEU so sánh n-gram của caption sinh ra với caption ground truth.
> BLEU-4 là khó nhất và quan trọng nhất — yêu cầu 4 từ liên tiếp phải đúng.
> Nhóm em đánh giá trên 1,000 ảnh test, mỗi ảnh 5 caption tham chiếu — tổng 5,000 câu — nên kết quả đáng tin cậy."

---

---

# SLIDE 15 — KẾT QUẢ SO SÁNH 3 MÔ HÌNH

## BLEU Score trên tập Test (1,000 ảnh, Beam Width = 5)

<br>

| Mô hình               | BLEU-1    | BLEU-2    | BLEU-3    | BLEU-4       |
| --------------------- | --------- | --------- | --------- | ------------ |
| VGG16 + LSTM          | 61.6%     | 43.6%     | 30.5%     | 21.3%        |
| CLIP + LSTM           | 66.0%     | 48.3%     | 34.4%     | 24.3%        |
| **ResNet-101 + LSTM** | **68.1%** | **49.5%** | **35.0%** | **24.8%** ✅ |

<br>

### So với baseline VGG16:

| Mô hình        | BLEU-4    | Cải thiện     |
| -------------- | --------- | ------------- |
| VGG16          | 21.3%     | —             |
| CLIP           | 24.3%     | **+14.1%**    |
| **ResNet-101** | **24.8%** | **+16.4%** 🏆 |

<br>

📌 **Hiển thị biểu đồ so sánh BLEU từ notebook**

---

> 🗣️ **:** "Đây là kết quả cốt lõi của đề tài.
> **ResNet-101 đạt BLEU-4 cao nhất: 24.8%**, cải thiện 16.4% so với VGG16.
> **CLIP đứng thứ hai: 24.3%** — cải thiện 14.1%.
> Điều bất ngờ là **CLIP không vượt qua được ResNet-101** dù lý thuyết CLIP mạnh hơn.
> Nhóm em sẽ phân tích nguyên nhân ngay sau đây."

---

---

# SLIDE 16 — DEMO: ẢNH THẬT VÀ CAPTION SINH RA

## Kết quả trực quan

📌 **Hiển thị Demo từ notebook — 4 ảnh test với caption từ 3 mô hình**

<br>

### Ví dụ caption tốt nhất (VGG16, BLEU-4 = 0.603):

```
Ảnh: [Người đàn ông và phụ nữ đang nhảy]
Caption sinh ra: "a man and a woman are dancing on a dance floor"
Ground truth:    "a man and a woman are dancing together"
```

<br>

### Ví dụ caption thất bại:

```
Ảnh: [Cảnh thiên nhiên phức tạp]
Caption sinh ra: "a man is standing in a field"  ← không có người
Ground truth:    "a mountain with snow on it"
```

> ⚠️ Mô hình gặp khó khăn với: cảnh thiên nhiên, nhiều người, hoạt động trừu tượng

---

> 🗣️ **:** "Đây là một số kết quả trực quan.
> Mô hình hoạt động tốt với ảnh đơn giản như người đang thực hiện một hành động rõ ràng.
> Trường hợp thất bại thường xảy ra với cảnh thiên nhiên, ảnh nhiều đối tượng phức tạp, hoặc hoạt động trừu tượng.
> Đây là hạn chế chung của kiến trúc LSTM — **không thể hiểu context toàn cục** của ảnh tốt bằng Transformer."

---

---

# SLIDE 17 — PHÂN TÍCH KẾT QUẢ

## Tại sao ResNet-101 thắng CLIP?

<br>

### Phân tích kỹ thuật:

| Yếu tố               | ResNet-101     | CLIP           |
| -------------------- | -------------- | -------------- |
| Feature dimension    | **2048 chiều** | 512 chiều      |
| Thông tin không gian | Phong phú hơn  | Đã nén qua ViT |
| Phù hợp với LSTM     | ✅ Cao         | Trung bình     |
| Epochs huấn luyện    | 10             | 10             |

<br>

### Lý giải:

> 1. **CLIP feature 512d** — nhỏ hơn ResNet 2048d → ít thông tin không gian hơn cho Attention
> 2. **CLIP tối ưu cho classification** (image-text matching), không phải generation
> 3. **Decoder LSTM không đủ sức** tận dụng hết sức mạnh semantic của CLIP
> 4. Nếu dùng **Transformer decoder** thay vì LSTM → CLIP có thể vượt ResNet

<br>

> ✅ Kết luận: ResNet-101 **phù hợp hơn** với kiến trúc LSTM + Attention trong bài toán này

---

> 🗣️ **:** "Câu hỏi thú vị nhất của đề tài là: Tại sao CLIP — mô hình mạnh hơn — lại không thắng ResNet-101?
> Chúng em phân tích và có 3 lý do chính.
> Thứ nhất, feature của ResNet-101 là **2048 chiều** — nhiều thông tin không gian hơn CLIP 512 chiều.
> Thứ hai, CLIP được tối ưu để **hiểu ngữ nghĩa chung** của ảnh, không phải trích xuất chi tiết từng vùng.
> Thứ ba, **LSTM không đủ mạnh** để khai thác hết sức mạnh semantic của CLIP.
> Nếu thay LSTM bằng Transformer decoder — đây là hướng phát triển tiếp theo — thì CLIP có thể sẽ vượt."

---

---

# SLIDE 18 — SO SÁNH VỚI NGHIÊN CỨU KHÁC

## Benchmarking — Kết quả của nhóm so với công bố

<br>

| Công trình                       | Dataset   | Encoder       | BLEU-4       |
| -------------------------------- | --------- | ------------- | ------------ |
| Xu et al., 2015 (Hard Attention) | Flickr30k | VGG16         | 19.9%        |
| Vinyals et al., 2015 (Google)    | Flickr30k | GoogLeNet     | 24.3%        |
| **Nhóm em — VGG16**              | Flickr30k | VGG16         | **21.3%**    |
| **Nhóm em — CLIP**               | Flickr30k | CLIP ViT-B/32 | **24.3%**    |
| **Nhóm em — ResNet-101**         | Flickr30k | ResNet-101    | **24.8%** 🏆 |

<br>

> ✅ Kết quả của nhóm **vượt Xu et al. (2015)** và **tương đương Google (2015)** chỉ với LSTM thuần
> ✅ Điều này xác nhận kiến trúc nhóm em xây dựng hoạt động **đúng và hiệu quả**

---

> 🗣️ **:** "Để đặt kết quả vào bối cảnh, nhóm em so sánh với các công trình nổi tiếng.
> Kết quả VGG16 của nhóm em là 21.3% — **vượt qua Xu et al. 2015** là 19.9%.
> Kết quả ResNet-101 là 24.8% — **ngang bằng Google 2015** dù chúng em chỉ dùng LSTM đơn giản.
> Điều này cho thấy implementation của nhóm em là **chính xác và không bị lỗi** — kết quả phản ánh đúng khả năng của mô hình."

---

---

# SLIDE 19 — KẾT LUẬN & HƯỚNG PHÁT TRIỂN

## Những gì nhóm em đã làm được:

✅ Xây dựng và so sánh thành công **3 mô hình** Image Captioning trên Flickr30k
✅ Kết quả **vượt baseline Xu et al. 2015**, tương đương Google 2015
✅ **ResNet-101 đạt BLEU-4 tốt nhất: 24.8%** — cải thiện 16.4% so với VGG16
✅ Phát hiện bất ngờ: **CLIP không vượt được ResNet với LSTM decoder**
✅ Xây dựng **Streamlit demo app** cho phép upload ảnh và sinh caption real-time

<br>

## Hướng phát triển tiếp theo:

| Hướng                                  | Kỳ vọng                                  |
| -------------------------------------- | ---------------------------------------- |
| Thay LSTM bằng **Transformer decoder** | CLIP có thể phát huy toàn bộ sức mạnh    |
| **Fine-tune encoder** thay vì frozen   | Đặc trưng tốt hơn cho bài toán cụ thể    |
| Dataset lớn hơn (COCO — 330K ảnh)      | BLEU-4 có thể đạt >30%                   |
| **Scheduled Sampling**                 | Giảm exposure bias — cải thiện inference |

---

> 🗣️ **:** "Tóm lại, nhóm em đã hoàn thành tất cả mục tiêu đề ra.
> Ba mô hình đều hoạt động tốt, kết quả vượt các nghiên cứu cùng kiến trúc từ 2015.
> Phát hiện thú vị nhất là CLIP không vượt ResNet với LSTM — điều này gợi mở hướng nghiên cứu tiếp theo.
> Nhóm em cũng đã xây dựng **demo app thực tế** — không chỉ là lý thuyết mà có sản phẩm cụ thể có thể dùng ngay."

---

---

# SLIDE 20 — CẢM ƠN & HỎI ĐÁP

<br>
<br>

# CẢM ƠN Thầy ĐÃ LẮNG NGHE

<br>

**Nhóm sinh viên thực hiện:**

> Lê Văn Cảnh &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp; Nguyễn Đức Trường Giang

**Giảng viên hướng dẫn:** Hồ Nhựt Minh

**Lớp:** DT2307L — Aptech

<br>
<br>

## Nhóm em xin sẵn sàng trả lời câu hỏi của Thầy

---

> 🗣️ **CẢ HAI NÓI:** "Nhóm em xin cảm ơn thầy Hồ Nhựt Minh đã hướng dẫn và hỗ trợ trong suốt quá trình thực hiện.
> Cảm ơn Thầy đã lắng nghe. Nhóm em xin sẵn sàng trả lời câu hỏi."

---

---

# PHỤ LỤC — CÂU HỎI THƯỜNG GẶP (CHUẨN BỊ TRƯỚC)

> ⚠️ Phần này KHÔNG trình chiếu — chỉ để nhóm chuẩn bị trả lời Thầy

---

## ❓ Q1: Tại sao ResNet-101 tốt hơn CLIP?

**Trả lời:** CLIP feature 512 chiều nhỏ hơn ResNet 2048 chiều, cung cấp ít thông tin không gian hơn cho cơ chế Attention. Ngoài ra, CLIP được huấn luyện cho image-text matching (phân loại), không phải generation — nên LSTM khó khai thác được sức mạnh ngữ nghĩa của CLIP. Để CLIP phát huy hết tiềm năng cần Transformer decoder thay vì LSTM.

---

## ❓ Q2: BLEU score có phải thước đo tốt nhất không?

**Trả lời:** BLEU là thước đo chuẩn và được dùng rộng rãi nhất (từ 2002, Papineni et al.), dễ so sánh với các nghiên cứu khác. Tuy nhiên BLEU có hạn chế: không đánh giá được tính trôi chảy ngữ nghĩa cao. Các thước đo tốt hơn như CIDEr, METEOR, SPICE tập trung hơn vào semantic — đây là hướng cải thiện đánh giá cho đề tài sau.

---

## ❓ Q3: Tại sao không fine-tune encoder?

**Trả lời:** Encoder được giữ frozen để tiết kiệm bộ nhớ GPU và tránh overfitting khi dataset không quá lớn (31K ảnh). Fine-tuning encoder cần GPU mạnh hơn và cần kỹ thuật discriminative fine-tuning để không làm hỏng pretrained weights. Đây là hướng phát triển tiếp theo có thể cải thiện BLEU thêm 1-2%.

---

## ❓ Q4: Tại sao VGG16 cần 30 epochs còn ResNet/CLIP chỉ cần 10?

**Trả lời:** Feature của ResNet-101 (2048d) và CLIP phong phú và biểu đạt hơn feature VGG16 (512d). Decoder LSTM học cách "đọc" feature tốt hơn trong ít epoch hơn. VGG16 cần nhiều epoch hơn để LSTM học cách bù đắp cho feature ít thông tin hơn.

---

## ❓ Q5: Beam Search width = 5 thì chọn như thế nào?

**Trả lời:** Beam width 5 là giá trị chuẩn trong hầu hết các paper Image Captioning (Xu et al. 2015, Vinyals et al. 2015). Width lớn hơn không nhất thiết tốt hơn vì có thể sinh ra câu quá dài, lan man. Nhóm em giữ width = 5 để kết quả có thể so sánh trực tiếp với các công trình trước.

---

## ❓ Q6: Dataset Flickr30k có đủ lớn không?

**Trả lời:** Flickr30k với 31,783 ảnh là dataset chuẩn, đủ để train mô hình có kết quả tốt và có thể so sánh với nhiều nghiên cứu. Dataset lớn hơn như COCO (330K ảnh) có thể cải thiện thêm BLEU-4 lên 30-35% — đây là hướng phát triển tiếp theo nếu có tài nguyên tính toán lớn hơn.

---

## ❓ Q7: Demo app hoạt động như thế nào?

**Trả lời:** Demo app được xây dựng bằng Streamlit. Người dùng upload ảnh → app trích xuất feature bằng encoder đã train → LSTM sinh caption bằng Beam Search → hiển thị caption và attention heatmap. Toàn bộ pipeline chạy trên CPU/MPS (Apple Silicon) trong vài giây.

---

## ❓ Q8: Nếu làm lại, nhóm em sẽ cải thiện gì?

**Trả lời:** Ba điểm chính: (1) Thay LSTM bằng **Transformer decoder** để tận dụng tốt hơn CLIP feature; (2) Dùng **CIDEr-D optimization** thay vì cross-entropy thuần; (3) Thử **Scheduled Sampling** trong training để giảm exposure bias — giúp inference tốt hơn khi không có teacher forcing.

---
