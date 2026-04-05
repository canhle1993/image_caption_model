# Neural Image Caption Generation

Đồ án nghiên cứu bài toán sinh mô tả ảnh tự động bằng học sâu, tập trung vào kiến trúc `Encoder-Decoder` với `Bahdanau Attention` và so sánh 3 encoder khác nhau trên bộ dữ liệu `Flickr30k`.

## Tổng Quan

Repo hiện tại gồm:

- 3 notebook chính cho 3 mô hình:
  - [ImageCaptioning_Flickr30k_VGG16.ipynb](./ImageCaptioning_Flickr30k_VGG16.ipynb)
  - [ImageCaptioning_Flickr30k_ResNet101.ipynb](./ImageCaptioning_Flickr30k_ResNet101.ipynb)
  - [ImageCaptioning_Flickr30k_CLIP.ipynb](./ImageCaptioning_Flickr30k_CLIP.ipynb)
- 1 notebook cũ theo hướng Flickr8k/VGG16 để tham khảo:
  - [ImageCaptioning_Flickr8k_VGG16_Keras.ipynb](./ImageCaptioning_Flickr8k_VGG16_Keras.ipynb)
- App demo Streamlit:
  - [streamlit2_app.py](./streamlit2_app.py)
  - [streamlit_app.py](./streamlit_app.py)
- Tài liệu báo cáo và slide bảo vệ:
  - [BAO_CAO_DO_AN.md](./BAO_CAO_DO_AN.md)
  - [SLIDE_BAO_VE.md](./SLIDE_BAO_VE.md)
  - [SLIDE_BAO_VE.html](./SLIDE_BAO_VE.html)
  - [BAO_VE_DO_AN.pptx](./BAO_VE_DO_AN.pptx)
- Script hỗ trợ sinh báo cáo và slide:
  - [generate_report.py](./generate_report.py)
  - [generate_slides.py](./generate_slides.py)
  - [generate_html_presentation.py](./generate_html_presentation.py)
- Bộ ảnh minh họa kết quả:
  - [images/CLIP](./images/CLIP)
  - [images/Resnet](./images/Resnet)
  - [images/VGG16](./images/VGG16)

## Bài Toán Và Kiến Trúc

Pipeline chính của đề tài:

1. Ảnh đầu vào được encoder trích xuất đặc trưng không gian dạng `(49, D)`.
2. Decoder `LSTM + Bahdanau Attention` sinh caption theo từng từ.
3. Huấn luyện với `Teacher Forcing`.
4. Suy diễn bằng `Beam Search`.
5. Đánh giá bằng `BLEU-1` đến `BLEU-4`.

Ba mô hình được so sánh công bằng vì dùng chung:

- dataset `Flickr30k`
- decoder `LSTM + Attention`
- vocabulary và GloVe initialization
- tập test 1,000 ảnh

## Kết Quả Chính

Kết quả trên tập test Flickr30k:

| Mô hình | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---|---|---|---|
| VGG16 + LSTM | 61.6% | 43.6% | 30.5% | 21.3% |
| CLIP + LSTM | 66.0% | 48.3% | 34.4% | 24.3% |
| ResNet-101 + LSTM | 68.1% | 49.5% | 35.0% | 24.8% |

Điểm nổi bật:

- `ResNet-101` cho kết quả tốt nhất trong 3 mô hình.
- `CLIP` rất mạnh về ngữ nghĩa nhưng chưa vượt `ResNet-101` khi kết hợp với decoder `LSTM`.
- Bộ slide và HTML presentation trong repo được dựng trực tiếp từ nội dung bảo vệ đồ án.

## Chạy Notebook

Môi trường khuyến nghị:

- macOS trên Apple Silicon
- Python 3.10+
- Jupyter hoặc VS Code Notebook
- PyTorch / TensorFlow tùy notebook hoặc script bạn đang mở

Cách chạy:

1. Clone repo.
2. Tạo môi trường Python phù hợp.
3. Cài các thư viện cần thiết theo notebook hoặc app bạn muốn chạy.
4. Mở một trong 3 notebook Flickr30k và chạy lần lượt từng cell.

## Chạy Demo Streamlit

Chạy app:

```bash
python streamlit2_app.py
```

Hoặc:

```bash
./run_streamlit.sh
```

## Trình Chiếu HTML

File [SLIDE_BAO_VE.html](./SLIDE_BAO_VE.html) có thể mở trực tiếp trên trình duyệt để thuyết trình.

Phím tắt:

- `←` `→` để chuyển slide
- `N` để bật / tắt lời thoại presenter notes
- `F` để fullscreen

Nếu chỉnh lại nội dung file markdown slide, có thể sinh lại HTML bằng:

```bash
python generate_html_presentation.py
```

## Dữ Liệu Và File Nặng

- File embedding lớn như `glove.6B.300d.txt` không nên commit lên GitHub vì vượt giới hạn file size.
- Datasets, checkpoints và artifact huấn luyện nên giữ local trong `workspace/`.
- Nếu cần tái lập thí nghiệm, hãy tự tải embedding và dataset về máy rồi đặt đúng vị trí.

## Ghi Chú

- Một số tài liệu trong repo phục vụ trực tiếp cho báo cáo và bảo vệ đồ án.
- Thư mục `images/` chứa các biểu đồ và demo caption đã xuất ra từ notebook để chèn vào slide.
- Repo hiện tập trung vào giá trị nghiên cứu, so sánh mô hình và trình bày kết quả hơn là đóng gói production.
