"""
Run this script ONCE to train and save the model + vectorizer.
Place train.txt in the same directory before running.

Usage:
    python save_model.py
"""

import pickle
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# ── Emotion label map (same order as your notebook) ──────────────────────────
EMOTION_LABELS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("train.txt", sep=";", header=None, names=["text", "emotion"])

unique_emotions = df["emotion"].unique()
emotion_numbers = {emo: i for i, emo in enumerate(unique_emotions)}
# Save the reverse mapping so the app can decode predictions
label_map = {v: k for k, v in emotion_numbers.items()}

df["emotion"] = df["emotion"].map(emotion_numbers)

# ── Preprocessing ─────────────────────────────────────────────────────────────
stop_words = set(stopwords.words("english"))


def preprocess(txt: str) -> str:
    txt = txt.lower()
    txt = txt.translate(str.maketrans("", "", string.punctuation))
    txt = "".join(c for c in txt if not c.isdigit())
    txt = "".join(c for c in txt if c.isascii())
    words = word_tokenize(txt)
    txt = " ".join(w for w in words if w not in stop_words)
    return txt


df["text"] = df["text"].apply(preprocess)

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["emotion"], test_size=0.20, random_state=42
)

# ── Vectorize + train ─────────────────────────────────────────────────────────
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=list(label_map.values())))

# ── Save artifacts ────────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("\n✅ Saved: model.pkl, vectorizer.pkl, label_map.pkl")
