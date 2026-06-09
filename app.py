import pickle
import string
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ── NLTK setup ────────────────────────────────────────────────────────────────
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🎭",
    layout="centered",
)

# ── Emotion metadata ──────────────────────────────────────────────────────────
EMOTION_META = {
    "sadness":  {"emoji": "😢", "color": "#5b8def"},
    "joy":      {"emoji": "😄", "color": "#f5c518"},
    "love":     {"emoji": "❤️",  "color": "#e05c97"},
    "anger":    {"emoji": "😠", "color": "#e05c3a"},
    "fear":     {"emoji": "😨", "color": "#8c6abf"},
    "surprise": {"emoji": "😲", "color": "#3abfa0"},
}

# ── Load model artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("label_map.pkl", "rb") as f:
        label_map = pickle.load(f)
    return model, vectorizer, label_map


try:
    model, vectorizer, label_map = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Preprocessing (mirrors your notebook exactly) ─────────────────────────────
stop_words = set(stopwords.words("english"))


def preprocess(txt: str) -> str:
    txt = txt.lower()
    txt = txt.translate(str.maketrans("", "", string.punctuation))
    txt = "".join(c for c in txt if not c.isdigit())
    txt = "".join(c for c in txt if c.isascii())
    words = word_tokenize(txt)
    txt = " ".join(w for w in words if w not in stop_words)
    return txt


def predict_emotion(text: str):
    cleaned = preprocess(text)
    vec = vectorizer.transform([cleaned])
    pred_idx = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    emotion_name = label_map[pred_idx]
    return emotion_name, proba, pred_idx


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='text-align:center; margin-bottom:0'>🎭 Emotion Detector</h1>
    <p style='text-align:center; color:gray; margin-top:4px'>
        Powered by TF-IDF + Logistic Regression · Kaggle Emotions Dataset
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

if not model_loaded:
    st.error(
        "⚠️ Model files not found! Run `python save_model.py` first to generate "
        "`model.pkl`, `vectorizer.pkl`, and `label_map.pkl`."
    )
    st.stop()

# ── Input area ────────────────────────────────────────────────────────────────
st.markdown("#### ✍️ Enter your text below")
user_input = st.text_area(
    label="",
    placeholder="Type something like: 'I feel so happy today!' or 'This makes me really nervous...'",
    height=140,
)

col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    predict_btn = st.button("🔍 Analyse", use_container_width=True, type="primary")

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    if not user_input.strip():
        st.warning("Please enter some text before clicking Analyse.")
    else:
        emotion, proba, pred_idx = predict_emotion(user_input)
        meta = EMOTION_META.get(emotion, {"emoji": "🤔", "color": "#888888"})

        st.markdown("---")
        st.markdown("#### 🎯 Result")

        # Main result card
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {meta['color']}22, {meta['color']}44);
                border-left: 6px solid {meta['color']};
                border-radius: 12px;
                padding: 20px 28px;
                margin-bottom: 20px;
            ">
                <p style="font-size:48px; margin:0; line-height:1">{meta['emoji']}</p>
                <h2 style="margin:8px 0 4px; text-transform:capitalize; color:{meta['color']}">
                    {emotion}
                </h2>
                <p style="color:gray; margin:0">Detected emotion</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Confidence bar chart for all emotions
        st.markdown("#### 📊 Confidence scores")
        classes = [label_map[i] for i in range(len(label_map))]

        for i, (cls, prob) in enumerate(zip(classes, proba)):
            m = EMOTION_META.get(cls, {"emoji": "🤔", "color": "#888"})
            bar_pct = prob * 100
            is_pred = cls == emotion
            weight = "700" if is_pred else "400"
            st.markdown(
                f"""
                <div style="margin-bottom:10px">
                    <div style="display:flex; justify-content:space-between; font-weight:{weight}">
                        <span>{m['emoji']} {cls.capitalize()}</span>
                        <span>{bar_pct:.1f}%</span>
                    </div>
                    <div style="background:#e0e0e0; border-radius:6px; height:10px; margin-top:4px">
                        <div style="
                            background:{m['color']};
                            width:{bar_pct}%;
                            height:10px;
                            border-radius:6px;
                            transition:width 0.4s ease;
                        "></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Show preprocessed text
        with st.expander("🔎 See preprocessed text"):
            st.code(preprocess(user_input), language=None)

# ── Try example sentences ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 💡 Try an example")

examples = [
    ("😄 Joy",      "I am so happy and excited about my results today!"),
    ("😢 Sadness",  "I miss my old friends and feel really lonely lately."),
    ("😠 Anger",    "This is absolutely unacceptable and makes me furious."),
    ("😨 Fear",     "I'm terrified of what might happen tomorrow at the exam."),
    ("❤️ Love",     "I love spending time with my family, they mean everything."),
    ("😲 Surprise", "I cannot believe she actually showed up! What a shock!"),
]

cols = st.columns(3)
for idx, (label, sentence) in enumerate(examples):
    with cols[idx % 3]:
        if st.button(label, key=f"ex_{idx}", use_container_width=True):
            st.session_state["example_text"] = sentence
            st.rerun()

# Populate textarea from example click
if "example_text" in st.session_state:
    txt = st.session_state.pop("example_text")
    emotion, proba, pred_idx = predict_emotion(txt)
    meta = EMOTION_META.get(emotion, {"emoji": "🤔", "color": "#888888"})

    st.info(f"**Input:** {txt}")
    st.markdown(
        f"""
        <div style="
            background: {meta['color']}22;
            border-left: 5px solid {meta['color']};
            border-radius: 10px;
            padding: 14px 20px;
        ">
            {meta['emoji']} <strong style="text-transform:capitalize">{emotion}</strong>
            &nbsp;|&nbsp; Confidence: <strong>{max(proba)*100:.1f}%</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:13px'>"
    "Built with Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
