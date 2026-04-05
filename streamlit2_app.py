"""
streamlit2_app.py — Demo app for the PyTorch CaptionDecoder model (Flickr30k)
Usage:
    /opt/anaconda3/envs/tf-metal/bin/python -m streamlit run streamlit2_app.py
"""

from pathlib import Path
import pickle
from typing import Optional

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.models as tv_models
import torchvision.transforms as transforms


# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent
WORK_DIR       = PROJECT_ROOT / "workspace"
MODEL_PATH     = WORK_DIR / "models" / "caption_flickr30k_best.pt"
TOKENIZER_PATH = WORK_DIR / "tokenizer_flickr30k.pkl"

NUM_REGIONS  = 49
FEATURE_SIZE = 512
IMG_SIZE     = 224

VGG_MEAN = [0.485, 0.456, 0.406]
VGG_STD  = [0.229, 0.224, 0.225]


# ── Model architecture (must match training code exactly) ──────────────────────
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int, feature_size: int):
        super().__init__()
        self.W_h = nn.Linear(hidden_size,  hidden_size, bias=False)
        self.W_f = nn.Linear(feature_size, hidden_size, bias=False)
        self.V   = nn.Linear(hidden_size,  1,           bias=False)

    def forward(self, hidden: torch.Tensor, features: torch.Tensor):
        h_exp   = self.W_h(hidden).unsqueeze(1)
        f_proj  = self.W_f(features)
        energy  = self.V(torch.tanh(h_exp + f_proj))
        weights = torch.softmax(energy, dim=1)
        context = (weights * features).sum(dim=1)
        return context, weights.squeeze(-1)


class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 feature_size: int = 512, dropout: float = 0.5,
                 embedding_matrix: Optional[np.ndarray] = None,
                 embed_trainable: bool = True):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(
                torch.FloatTensor(embedding_matrix),
                requires_grad=embed_trainable,
            )

        self.attention = BahdanauAttention(hidden_size, feature_size)
        self.lstm    = nn.LSTM(embed_dim + feature_size, hidden_size, batch_first=True)
        self.init_h  = nn.Linear(feature_size, hidden_size)
        self.init_c  = nn.Linear(feature_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc_out  = nn.Linear(hidden_size, vocab_size)

    def _init_hidden(self, features: torch.Tensor):
        mean_feat = features.mean(dim=1)
        h = torch.tanh(self.init_h(mean_feat)).unsqueeze(0)
        c = torch.tanh(self.init_c(mean_feat)).unsqueeze(0)
        return h, c

    def forward(self, features: torch.Tensor, captions: torch.Tensor):
        emb  = self.dropout(self.embedding(captions))
        h, c = self._init_hidden(features)
        logits = []
        for t in range(captions.size(1)):
            h_t         = h.squeeze(0)
            context, _  = self.attention(h_t, features)
            lstm_in     = torch.cat([emb[:, t, :], context], dim=1).unsqueeze(1)
            out, (h, c) = self.lstm(lstm_in, (h, c))
            logit       = self.fc_out(self.dropout(out.squeeze(1)))
            logits.append(logit)
        return torch.stack(logits, dim=1)


# ── Beam search (standalone, no global deps) ──────────────────────────────────
@torch.no_grad()
def beam_search(
    model: CaptionDecoder,
    features: torch.Tensor,
    word2idx: dict,
    idx2word: dict,
    beam_width: int = 5,
    max_len: int = 35,
) -> str:
    PAD_ID   = 0
    START_ID = word2idx.get("startseq", 1)
    END_ID   = word2idx.get("endseq", 2)
    UNK_ID   = word2idx.get("<unk>", word2idx.get("unk", -1))
    skip_ids = {PAD_ID, START_ID, END_ID, UNK_ID}

    model.eval()
    device = features.device
    h0, c0 = model._init_hidden(features)
    beams   = [(0.0, [START_ID], h0, c0)]
    completed = []

    for _ in range(max_len):
        candidates = []
        for score, tokens, bh, bc in beams:
            if tokens[-1] == END_ID:
                completed.append((score / len(tokens), tokens))
                continue
            last_t = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
            emb    = model.embedding(last_t)
            ctx, _ = model.attention(bh.squeeze(0), features)
            inp    = torch.cat([emb.squeeze(1), ctx], dim=1).unsqueeze(1)
            out, (new_h, new_c) = model.lstm(inp, (bh, bc))
            log_p  = F.log_softmax(model.fc_out(out.squeeze(1)), dim=-1)
            topk   = torch.topk(log_p, beam_width, dim=-1)
            for i in range(beam_width):
                tok    = topk.indices[0, i].item()
                new_sc = score + topk.values[0, i].item()
                candidates.append((new_sc, tokens + [tok], new_h, new_c))

        if not candidates:
            break
        candidates.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
        beams = candidates[:beam_width]

    for score, tokens, bh, bc in beams:
        completed.append((score / len(tokens), tokens))

    if not completed:
        return ""
    _, best_tokens = max(completed, key=lambda x: x[0])
    words = [
        idx2word.get(t, "")
        for t in best_tokens
        if t not in skip_ids and idx2word.get(t, "")
    ]
    return " ".join(words)


# ── Cached resource loaders ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@st.cache_resource(show_spinner=False)
def load_tokenizer() -> dict:
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy tokenizer tại {TOKENIZER_PATH}")
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_caption_model() -> CaptionDecoder:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model tại {MODEL_PATH}")
    tok    = load_tokenizer()
    device = get_device()
    model  = CaptionDecoder(
        vocab_size    = tok["vocab_size"],
        embed_dim     = tok["embed_dim"],
        hidden_size   = 512,
        feature_size  = FEATURE_SIZE,
        dropout       = 0.5,
    ).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_feature_extractor() -> nn.Module:
    device  = get_device()
    vgg16   = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
    extractor = vgg16.features.to(device).eval()
    for p in extractor.parameters():
        p.requires_grad = False
    return extractor


# ── Feature extraction ─────────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=VGG_MEAN, std=VGG_STD),
])


def extract_features(image: Image.Image, extractor: nn.Module) -> torch.Tensor:
    device = get_device()
    tensor = _transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = extractor(tensor)                      # (1, 512, 7, 7)
    feat = feat.squeeze(0).permute(1, 2, 0)          # (7, 7, 512)
    feat = feat.reshape(NUM_REGIONS, FEATURE_SIZE)    # (49, 512)
    return feat.unsqueeze(0)                          # (1, 49, 512)


# ── Main UI ────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Image Caption Generator (PyTorch)",
        page_icon="🖼️",
        layout="centered",
    )
    st.title("Image Caption Generator")
    st.caption("VGG16 conv features (49×512) + LSTM + Bahdanau Attention + GloVe 300d — Flickr30k")

    # Sidebar
    with st.sidebar:
        st.subheader("Model info")
        st.write(f"Checkpoint: `{MODEL_PATH.name}`")
        st.write(f"Tokenizer: `{TOKENIZER_PATH.name}`")
        beam_width = st.slider("Beam width", min_value=1, max_value=7, value=5, step=1)
        max_len    = st.slider("Max caption length", min_value=10, max_value=50, value=35, step=5)
        device     = get_device()
        st.write(f"Device: `{device}`")

        missing = []
        if not MODEL_PATH.exists():
            missing.append(str(MODEL_PATH))
        if not TOKENIZER_PATH.exists():
            missing.append(str(TOKENIZER_PATH))
        if missing:
            st.error("File không tồn tại:\n" + "\n".join(missing))

    # File uploader
    uploaded = st.file_uploader(
        "Upload ảnh",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded is None:
        st.info("Chọn một ảnh để sinh caption.")
        return

    image = Image.open(uploaded)
    st.image(image, caption=uploaded.name, use_container_width=True)

    if st.button("Generate Caption", type="primary", use_container_width=True):
        try:
            with st.spinner("Đang load model và trích xuất đặc trưng ảnh..."):
                tok       = load_tokenizer()
                model     = load_caption_model()
                extractor = load_feature_extractor()
                features  = extract_features(image, extractor)

            with st.spinner("Đang sinh caption (beam search)..."):
                caption = beam_search(
                    model,
                    features,
                    word2idx   = tok["word2idx"],
                    idx2word   = tok["idx2word"],
                    beam_width = beam_width,
                    max_len    = max_len,
                )

            st.subheader("Predicted Caption")
            st.success(caption if caption else "Không sinh được caption.")

            with st.expander("Chi tiết kỹ thuật"):
                st.write(f"Feature shape: `{tuple(features.shape)}`")
                st.write(f"Vocab size: `{tok['vocab_size']:,}`")
                st.write(f"Embed dim: `{tok['embed_dim']}`")
                st.write(f"Beam width: `{beam_width}`")
                st.write(f"Max length: `{max_len}`")
                st.write(f"Device: `{get_device()}`")

        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Lỗi: {e}")
            raise


if __name__ == "__main__":
    main()
