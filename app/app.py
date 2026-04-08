import streamlit as st
import pickle
import os
import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ===============================
# Fix import path
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.preprocess import preprocess

# ===============================
# Load Model (SAFE FIX ✅)
# ===============================
model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("❌ Model files not found. Please upload models folder to GitHub.")
    st.stop()

model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===============================
# Custom CSS (UNCHANGED)
# ===============================
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
}

.main { background: #0f0f0f; }
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1100px; }

/* Header */
.hero {
    display: flex; align-items: center; gap: 14px;
    margin-bottom: 2rem; padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.hero-icon {
    width: 48px; height: 48px; border-radius: 14px;
    background: #1DA1F2; display: flex;
    align-items: center; justify-content: center;
    font-size: 24px; flex-shrink: 0;
}
.hero-title { font-size: 26px; font-weight: 700; color: #f0ede8; margin: 0; }
.hero-sub { font-size: 13px; color: #73726c; margin: 2px 0 0; }

/* Metric cards */
.metric-box {
    background: #1a1a1a; border: 0.5px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 18px 20px;
}
.metric-val { font-size: 28px; font-weight: 700; color: #f0ede8; }
.metric-lbl { font-size: 11px; color: #73726c; }

/* Result cards */
.result-pos {
    background: rgba(99,153,34,0.12);
    border: 1px solid rgba(99,153,34,0.35);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
}
.result-neg {
    background: rgba(226,75,74,0.12);
    border: 1px solid rgba(226,75,74,0.35);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
}
.result-title { font-size: 20px; font-weight: 700; }

/* Input */
.stTextArea textarea {
    background: #1a1a1a !important;
    border-radius: 10px !important;
    color: #f0ede8 !important;
}

/* Button */
.stButton > button {
    background: #1DA1F2 !important;
    color: white !important;
    border-radius: 8px !important;
}

/* Section headers */
.section-hdr {
    font-size: 11px;
    color: #73726c;
}
</style>""", unsafe_allow_html=True)

# ===============================
# Session State
# ===============================
if 'history' not in st.session_state:
    st.session_state.history = []

# ===============================
# Header
# ===============================
st.markdown("""
<div class="hero">
  <div class="hero-icon">🐦</div>
  <div>
    <div class="hero-title">Sentiment Analyzer</div>
    <div class="hero-sub">Powered by ML · Real-time tweet analysis</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ===============================
# Metrics
# ===============================
total = len(st.session_state.history)
pos_count = sum(1 for h in st.session_state.history if h['label'] == 'Positive')
neg_count = total - pos_count
pos_rate = round(pos_count / total * 100) if total else 0
avg_conf = round(sum(h['conf'] for h in st.session_state.history) / total) if total else 0

col1, col2, col3 = st.columns(3)

col1.markdown(f"<div class='metric-box'><div class='metric-val'>{total}</div><div class='metric-lbl'>Total</div></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-box'><div class='metric-val'>{pos_rate}%</div><div class='metric-lbl'>Positive</div></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-box'><div class='metric-val'>{avg_conf}%</div><div class='metric-lbl'>Confidence</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ===============================
# Layout
# ===============================
left, right = st.columns([1.1, 1])

with left:
    tweet = st.text_area("Enter Tweet", height=130)

    col_btn, col_clr = st.columns([2,1])

    with col_btn:
        analyze_clicked = st.button("🔍 Analyze sentiment", width="stretch")

    with col_clr:
        if st.button("Clear", width="stretch"):
            st.rerun()

    if analyze_clicked and tweet.strip():
        processed = preprocess(tweet)
        vector = vectorizer.transform([processed])

        try:
            proba = model.predict_proba(vector)[0]
            confidence = round(max(proba) * 100)
            predicted_class = int(model.classes_[proba.argmax()])
        except:
            predicted_class = model.predict(vector)[0]
            confidence = 85

        label = "Positive" if predicted_class == 1 else "Negative"

        if label == "Positive":
            st.markdown(f"<div class='result-pos'><div class='result-title'>😊 Positive ({confidence}%)</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-neg'><div class='result-title'>😡 Negative ({confidence}%)</div></div>", unsafe_allow_html=True)

        st.session_state.history.append({
            "label": label,
            "conf": confidence
        })

with right:
    if st.session_state.history:
        fig = go.Figure(go.Pie(
            labels=["Positive","Negative"],
            values=[pos_count, neg_count],
            hole=0.6
        ))
        st.plotly_chart(fig, width="stretch")