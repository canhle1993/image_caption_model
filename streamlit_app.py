from pathlib import Path
from pickle import load

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


PROJECT_ROOT = Path(__file__).resolve().parent
WORK_DIR = PROJECT_ROOT / "workspace"
MODEL_PATH = WORK_DIR / "models" / "caption_model_attention.keras"
TOKENIZER_PATH = WORK_DIR / "tokenizer.pkl"
MAX_LENGTH = 34


def normalize_photo_feature(photo_feature: np.ndarray) -> np.ndarray:
    photo_feature = np.asarray(photo_feature, dtype="float32")
    if photo_feature.ndim == 1:
        norm = np.linalg.norm(photo_feature)
        if norm > 0:
            photo_feature = photo_feature / norm
    elif photo_feature.ndim == 2:
        norms = np.linalg.norm(photo_feature, axis=1, keepdims=True)
        photo_feature = photo_feature / np.maximum(norms, 1e-8)
    return photo_feature


@st.cache_resource(show_spinner=False)
def load_caption_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay model tai {MODEL_PATH}")
    return load_model(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_tokenizer():
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay tokenizer tai {TOKENIZER_PATH}")
    with open(TOKENIZER_PATH, "rb") as tokenizer_file:
        return load(tokenizer_file)


@st.cache_resource(show_spinner=False)
def load_feature_extractor():
    base_model = VGG16(weights="imagenet")
    return Model(
        inputs=base_model.inputs,
        outputs=base_model.get_layer("block5_pool").output,
        name="vgg16_conv_feature_extractor",
    )


def extract_conv_features(image: Image.Image, feature_extractor: Model) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224))
    image_array = np.asarray(image, dtype="float32")
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    conv_feature = feature_extractor.predict(image_array, verbose=0)
    conv_feature = conv_feature.reshape(
        conv_feature.shape[1] * conv_feature.shape[2],
        conv_feature.shape[3],
    )
    conv_feature = normalize_photo_feature(conv_feature)
    return conv_feature.astype("float32")


def word_for_id(integer: int, tokenizer) -> str | None:
    if integer <= 0:
        return None
    word = tokenizer.index_word.get(integer)
    if word is None:
        return None
    if tokenizer.num_words is not None and integer >= tokenizer.num_words:
        return None
    return word


def generate_caption_beam_search(
    model,
    tokenizer,
    photo_feature: np.ndarray,
    max_length: int,
    beam_width: int = 5,
    repetition_penalty: int = 2,
) -> str:
    normalized_feature = normalize_photo_feature(photo_feature)
    if normalized_feature.ndim == 2:
        normalized_feature = normalized_feature.reshape(
            1, normalized_feature.shape[0], normalized_feature.shape[1]
        )
    normalized_feature = normalized_feature.astype("float32")

    sequences = [("startseq", 0.0)]

    for _ in range(max_length):
        all_candidates = []
        for seq_text, score in sequences:
            last_tokens = seq_text.split()
            if last_tokens[-1] == "endseq":
                all_candidates.append((seq_text, score))
                continue

            encoded = tokenizer.texts_to_sequences([seq_text])[0]
            encoded = pad_sequences([encoded], maxlen=max_length, padding="post").astype(
                "int32"
            )
            yhat = model.predict([normalized_feature, encoded], verbose=0)[0]

            candidate_ids = np.argsort(yhat)[-beam_width * 4 :][::-1]
            added = 0
            for token_id in candidate_ids:
                word = word_for_id(int(token_id), tokenizer)
                if not word:
                    continue

                penalty = 0.0
                if len(last_tokens) >= repetition_penalty and all(
                    tok == word for tok in last_tokens[-repetition_penalty:]
                ):
                    penalty = 5.0

                prob = float(yhat[token_id])
                candidate_score = score - np.log(prob + 1e-10) + penalty
                all_candidates.append((seq_text + " " + word, candidate_score))
                added += 1
                if added >= beam_width:
                    break

        sequences = sorted(all_candidates, key=lambda tup: tup[1])[:beam_width]
        if all(seq.split()[-1] == "endseq" for seq, _ in sequences):
            break

    best_sequence = sequences[0][0]
    final_tokens = []
    seen_repeats = 0
    prev_word = None
    for word in best_sequence.split():
        if word in ("startseq", "endseq"):
            continue
        if word == prev_word:
            seen_repeats += 1
            if seen_repeats >= 2:
                continue
        else:
            seen_repeats = 0
        final_tokens.append(word)
        prev_word = word

    return " ".join(final_tokens).strip()


def main():
    st.set_page_config(page_title="Image Caption Generator", page_icon="🖼️", layout="centered")
    st.title("Image Caption Generator")
    st.caption("VGG16 conv features + Attention Decoder + GloVe")

    with st.sidebar:
        st.subheader("Model")
        st.write(f"Model: `{MODEL_PATH.name}`")
        st.write(f"Tokenizer: `{TOKENIZER_PATH.name}`")
        beam_width = st.slider("Beam width", min_value=1, max_value=7, value=5, step=1)
        repetition_penalty = st.slider(
            "Repetition penalty window", min_value=1, max_value=4, value=2, step=1
        )
        gpu_devices = tf.config.list_physical_devices("GPU")
        st.write("GPU devices:", gpu_devices if gpu_devices else "CPU only")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Chon mot anh de sinh caption.")
        return

    image = Image.open(uploaded_file)
    st.image(image, caption=uploaded_file.name, use_container_width=True)

    if st.button("Generate caption", type="primary", use_container_width=True):
        with st.spinner("Dang load model va trich xuat dac trung anh..."):
            caption_model = load_caption_model()
            tokenizer = load_tokenizer()
            feature_extractor = load_feature_extractor()
            conv_features = extract_conv_features(image, feature_extractor)

        with st.spinner("Dang sinh caption..."):
            caption = generate_caption_beam_search(
                caption_model,
                tokenizer,
                conv_features,
                max_length=MAX_LENGTH,
                beam_width=beam_width,
                repetition_penalty=repetition_penalty,
            )

        st.subheader("Predicted caption")
        st.success(caption if caption else "Khong sinh duoc caption.")

        with st.expander("Technical details"):
            st.write(f"Feature shape: `{conv_features.shape}`")
            st.write(f"Max length: `{MAX_LENGTH}`")
            st.write(f"Beam width: `{beam_width}`")
            st.write(f"Repetition penalty window: `{repetition_penalty}`")


if __name__ == "__main__":
    main()
