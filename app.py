import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Bidirectional, Dense

st.set_page_config(page_title="Twitter Sentiment (Demo)", page_icon="💬")

st.title("Twitter Sentiment Analysis (BiLSTM demo)")
st.caption("Loads a saved model if available; trains a small demo model otherwise. Replace the sample CSVs with your data.")

DATA_PATH = Path(__file__).parent
MODEL_DIR = DATA_PATH / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "bilstm_demo.h5"

def basic_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\\S+|www\\S+", "", text)
    text = re.sub(r"[^a-z0-9\\s#@'’]+", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH / "twitter_training.csv")
    df["text"] = df["text"].astype(str).map(basic_clean)
    return df

@st.cache_resource(show_spinner=False)
def get_demo_model(vocab_size: int, maxlen: int, num_classes: int):
    if MODEL_PATH.exists():
        return load_model(MODEL_PATH.as_posix())

    model = Sequential([
        Embedding(vocab_size, 64, input_length=maxlen),
        Bidirectional(LSTM(32)),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def prepare(df):
    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["text"])
    X = tokenizer.texts_to_sequences(df["text"])
    X = pad_sequences(X, maxlen=40, padding="post", truncating="post")
    return X, y, tokenizer, le

df = load_data()
X, y, tokenizer, le = prepare(df)

model = get_demo_model(vocab_size=5000, maxlen=40, num_classes=len(le.classes_))

if not MODEL_PATH.exists():
    st.info("No saved model found. Training a tiny demo model (very quick) ...")
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=3, batch_size=16, verbose=0)
    model.save(MODEL_PATH.as_posix())
    st.success("Model trained and saved!")

with st.sidebar:
    st.header("Predict")
    user_text = st.text_area("Enter a tweet/text:", height=120, placeholder="Type something...")

    if st.button("Analyze"):
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            seq = tokenizer.texts_to_sequences([basic_clean(user_text)])
            seq = pad_sequences(seq, maxlen=40, padding="post", truncating="post")
            probs = model.predict(seq, verbose=0)[0]
            idx = int(np.argmax(probs))
            st.subheader("Prediction")
            st.write(f"**Label:** {le.classes_[idx]}")
            st.write("**Probabilities:**")
            st.json({cls: float(p) for cls, p in zip(le.classes_, probs)})

st.subheader("Dataset Preview")
st.dataframe(df.head(10))
