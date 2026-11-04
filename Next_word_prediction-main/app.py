import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------------------------
# Files expected in the same folder as app.py:
#   lstm_model.h5
#   tokenizer.pkl
#   max_len.pkl
# -------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "lstm_model.h5"
TOKENIZER_PATH = APP_DIR / "tokenizer.pkl"
MAX_LEN_PATH = APP_DIR / "max_len.pkl"


@st.cache_resource
def load_artifacts():
    missing = [
        path.name
        for path in (MODEL_PATH, TOKENIZER_PATH, MAX_LEN_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s): "
            + ", ".join(missing)
            + ". Put them in the same folder as app.py."
        )

    # compile=False is sufficient for inference and avoids requiring
    # the original training loss/optimizer configuration.
    model = load_model(MODEL_PATH, compile=False)

    with TOKENIZER_PATH.open("rb") as file:
        tokenizer = pickle.load(file)

    with MAX_LEN_PATH.open("rb") as file:
        max_len = int(pickle.load(file))

    # The model's output width is the number of word IDs it can predict.
    output_vocab_size = int(model.output_shape[-1])

    # Keep only words whose IDs correspond to an output neuron.
    index_to_word = {
        index: word
        for word, index in tokenizer.word_index.items()
        if 0 < index < output_vocab_size
    }

    # If the model has a fixed input width, use it as a consistency check.
    model_input_len = model.input_shape[1]
    if model_input_len is not None and int(model_input_len) != max_len:
        raise ValueError(
            f"Input length mismatch: max_len.pkl contains {max_len}, "
            f"but the model expects {model_input_len}. "
            "Use the max_len.pkl saved alongside this exact model."
        )

    return model, tokenizer, max_len, index_to_word


def predict_next_word(model, tokenizer, index_to_word, text, max_len):
    token_ids = tokenizer.texts_to_sequences([text])[0]

    if not token_ids:
        return None

    padded = pad_sequences(
        [token_ids],
        maxlen=max_len,
        padding="pre",
        truncating="pre",
    )

    probabilities = model.predict(padded, verbose=0)[0]
    predicted_id = int(np.argmax(probabilities))

    return index_to_word.get(predicted_id)


def generate_text(model, tokenizer, index_to_word, seed_text, max_len, words_to_generate):
    result = seed_text.strip()

    for _ in range(words_to_generate):
        next_word = predict_next_word(
            model, tokenizer, index_to_word, result, max_len
        )
        if not next_word:
            break
        result += " " + next_word

    return result


st.set_page_config(
    page_title="LSTM Next Word Predictor",
    page_icon="✍️",
    layout="centered",
)

st.title("✍️ LSTM Next Word Predictor")
st.write(
    "Enter a starting phrase and let your trained LSTM model predict "
    "the next word or continue the text."
)

try:
    model, tokenizer, max_len, index_to_word = load_artifacts()
except Exception as error:
    st.error(str(error))
    st.info(
        "Required files: lstm_model.h5, tokenizer.pkl, and max_len.pkl. "
        "They must be in the same directory as app.py."
    )
    st.stop()

with st.sidebar:
    st.subheader("Model information")
    st.write(f"**Input sequence length:** {max_len}")
    st.write(f"**Output vocabulary size:** {model.output_shape[-1]}")
    st.caption("The tokenizer and max_len must be the same ones used during training.")

seed_text = st.text_area(
    "Starting text",
    value="what are you",
    height=100,
    placeholder="Type a few words...",
)

number_of_words = st.slider(
    "Number of words to generate",
    min_value=1,
    max_value=30,
    value=10,
)

col1, col2 = st.columns(2)

with col1:
    predict_button = st.button("Predict next word", use_container_width=True)

with col2:
    generate_button = st.button(
        "Generate text", type="primary", use_container_width=True
    )

if predict_button or generate_button:
    if not seed_text.strip():
        st.warning("Please enter some starting text.")
    else:
        token_ids = tokenizer.texts_to_sequences([seed_text])[0]

        if not token_ids:
            st.warning(
                "The tokenizer could not recognize any words in that input. "
                "Try words that occur in your training dataset."
            )
        elif predict_button:
            next_word = predict_next_word(
                model, tokenizer, index_to_word, seed_text, max_len
            )
            if next_word:
                st.success(f"Predicted next word: **{next_word}**")
            else:
                st.warning("The model returned a word ID not found in the tokenizer.")
        else:
            generated = generate_text(
                model,
                tokenizer,
                index_to_word,
                seed_text,
                max_len,
                number_of_words,
            )
            st.subheader("Generated text")
            st.write(generated)
            st.caption(
                "Text is generated one word at a time using greedy decoding "
                "(the highest-probability next word)."
            )

st.divider()
st.caption("Demo interface for your trained LSTM next-word prediction model.")
