# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **Python**: 3.10 via Conda env `tf-metal` at `/opt/anaconda3/envs/tf-metal/`
- **Hardware**: macOS Apple Silicon with MPS (Metal Performance Shaders) via PyTorch
- **Framework**: PyTorch (migrated from TensorFlow/Keras)

## Running the project

**Notebook** (primary workflow): Open `Neural_Image_Caption_Generation.ipynb` in VS Code, select the `tf-metal` kernel, and run cells in order from Cell 1.

**Streamlit demo app**:
```bash
./run_streamlit.sh
# or directly:
/opt/anaconda3/envs/tf-metal/bin/python -m streamlit run streamlit_app.py
```

The Streamlit app requires trained artifacts under `workspace/models/`:
- `workspace/models/caption_flickr30k_best.pt` — trained PyTorch model checkpoint
- `workspace/tokenizer_flickr30k.pkl` — fitted tokenizer (word2idx, idx2word, vocab_size, max_len, embed_dim)

## Architecture

This is a single-notebook ML project with a companion Streamlit app. There are no Python modules outside `streamlit_app.py`.

### Notebook cell structure

| Cell | Title | Description |
|------|-------|-------------|
| 1 | Setup & Environment | Project root setup, directory checks |
| 2 | Package Check & Device | Install packages, detect CPU/MPS/CUDA |
| 3 | Import Libraries | All imports (PyTorch, torchvision, nltk, matplotlib, etc.) |
| 4 | Config & Constants | All hyperparameters and file paths |
| 5 | EDA | Image size analysis, caption statistics, vocab analysis |
| 6 | EDA (code) | Flickr30k dataset exploration with 4 plot groups |
| 7 | VGG16 Feature Extraction | Extract & cache `(49, 512)` conv features |
| 8 | Parse Captions | Read `results.csv`, build train/val/test splits, save JSON |
| 9 | Caption Cleaning | Clean captions, add `startseq`/`endseq`, keep single-char words (e.g. "a", "i") |
| 10 | Tokenization & GloVe | Build vocab, load GloVe 300d, build embedding matrix |
| 11 | Dataset & DataLoader | `CaptionDataset` class, teacher forcing setup |
| 12 | Model Architecture | `CaptionDecoder` + `BahdanauAttention` class definitions |
| 13 | Training | Training loop with early stopping, save best checkpoint |
| 14 | BLEU Evaluation | Beam search on test set, compute BLEU-1 to BLEU-4 |
| 15 | Demo | Visual caption generation demo, error analysis |

### Data pipeline

1. **Flickr30k dataset** in `workspace/Flicker30k_Dataset/` (not committed): ~31,783 images
2. **Captions**: `workspace/results.csv` (pipe-separated: `image_name|comment_number|comment`)
3. **VGG16 feature extraction** (PyTorch, `conv5_3` layer):
   - `workspace/features_conv_flickr30k.pkl` — `{image_id: np.ndarray (49, 512)}`
4. **Caption cleaning** (`clean_caption`): lowercase, remove non-alpha, keep words `len >= 1` (preserves "a", "i"), add `startseq`/`endseq`, saved to `workspace/descriptions_flickr30k.txt`
5. **Train/Val/Test splits**: saved to `workspace/flickr30k_splits.json` (seed=42, val=1014, test=1000)
6. **GloVe embeddings**: `glove.6B.300d.txt` in project root (not committed); 99.2% vocab coverage

### Model

- **Class**: `CaptionDecoder` (LSTM-512 + Bahdanau Attention), ~11.9M parameters
- **Inputs**: image conv features `(49, 512)` + partial caption token sequence `(MAX_LEN,)`
- **Image branch**: VGG16 `conv5_3` features — 49 spatial positions × 512 channels
- **Text branch**: GloVe-initialized Embedding (300d, fine-tunable) → LSTMCell
- **Attention**: `BahdanauAttention` — additive attention over LSTM hidden state and image features
- **Output**: Linear → softmax over vocab (size 10,000)
- **Loss**: CrossEntropyLoss with `ignore_index=<pad>`, `label_smoothing=0.1`
- **Optimizer**: AdamW, `lr=1e-4`, `weight_decay=1e-4`
- **Scheduler**: ReduceLROnPlateau, `factor=0.5`, `patience=3`
- **Early stopping**: patience=7

### Inference

- **Beam search** (`generate_beam` method on `CaptionDecoder`): `beam_width=5`, `max_len=35`

### Evaluation (baseline from old Flickr8k model)

- BLEU-1 through BLEU-4 via `nltk.translate.bleu_score.corpus_bleu` on test split
- Flickr8k baseline: BLEU-1 0.5285, BLEU-2 0.3385, BLEU-3 0.2097, BLEU-4 0.1186

## Key paths

| Path | Purpose |
|------|---------|
| `workspace/Flicker30k_Dataset/` | Flickr30k images (not committed) |
| `workspace/results.csv` | Flickr30k captions (pipe-separated) |
| `workspace/features_conv_flickr30k.pkl` | Cached VGG16 conv features `(49, 512)` |
| `workspace/descriptions_flickr30k.txt` | Cleaned captions with startseq/endseq |
| `workspace/flickr30k_splits.json` | Train/val/test image ID splits |
| `workspace/tokenizer_flickr30k.pkl` | word2idx, idx2word, vocab metadata |
| `workspace/models/caption_flickr30k_best.pt` | Best PyTorch checkpoint |
| `workspace/models/training_history_flickr30k.json` | Training loss history |
| `workspace/models/model_architecture.json` | Model config snapshot |
| `glove.6B.300d.txt` | GloVe embeddings (not committed) |

## Key config flags

In cell 4, `FORCE_RETRAIN = False` controls whether training/feature extraction reruns even if a checkpoint exists. Set to `True` to force retraining from scratch.

## Hyperparameters (cell 4)

| Parameter | Value |
|-----------|-------|
| `VOCAB_SIZE` | 10,000 |
| `MAX_LENGTH` | 35 |
| `EMBED_DIM` | 300 (GloVe 300d) |
| `HIDDEN_SIZE` | 512 |
| `DROPOUT` | 0.5 |
| `BATCH_SIZE` | 64 |
| `EPOCHS` | 50 |
| `LEARNING_RATE` | 1e-4 |
| `BEAM_WIDTH` | 5 |
| `N_VAL` | 1,014 |
| `N_TEST` | 1,000 |
