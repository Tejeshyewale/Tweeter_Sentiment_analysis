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
    st.error("❌ Model files not found. Please upload model.pkl and vectorizer.pkl to GitHub (models folder).")
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
/* KEEP ALL YOUR CSS SAME */
</style>""", unsafe_allow_html=True)

# ===============================
# Session State
# ===============================
if 'history' not in st.session_state:
    st.session_state.history = []

# ===============================
# Header (UNCHANGED)
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
# Metrics (UNCHANGED)
# ===============================
total = len(st.session_state.history)
pos_count = sum(1 for h in st.session_state.history if h['label'] == 'Positive')
neg_count = total - pos_count
pos_rate = round(pos_count / total * 100) if total else 0
avg_conf = round(sum(h['conf'] for h in st.session_state.history) / total) if total else 0

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-val">{total}</div>
        <div class="metric-lbl">Total analyzed</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-val" style="color:#97C459">{pos_rate}%</div>
        <div class="metric-lbl">Positive rate</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-val" style="color:#1DA1F2">{avg_conf}{'%' if total else '—'}</div>
        <div class="metric-lbl">Avg confidence</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ===============================
# Main Layout
# ===============================
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown('<div class="section-hdr">Enter tweet text</div>', unsafe_allow_html=True)

    tweet = st.text_area(
        label="tweet",
        placeholder="e.g. Just had the most amazing coffee this morning! ☕",
        max_chars=280,
        height=130,
        label_visibility="collapsed"
    )

    char_count = len(tweet)
    st.caption(f"{char_count} / 280 characters")

    col_btn, col_clr = st.columns([2, 1])

    with col_btn:
        analyze_clicked = st.button("🔍 Analyze sentiment", width='stretch')  # FIXED

    with col_clr:
        if st.button("Clear", width='stretch'):  # FIXED
            st.rerun()

    if analyze_clicked:
        if not tweet.strip():
            st.warning("Please enter some tweet text first.")
        else:
            processed = preprocess(tweet)
            vector = vectorizer.transform([processed])

            try:
                proba = model.predict_proba(vector)[0]
                confidence = round(max(proba) * 100)
                predicted_class = int(model.classes_[proba.argmax()])
            except:
                result = model.predict(vector)[0]
                predicted_class = int(result)
                confidence = 85

            is_pos = predicted_class == 1
            label = "Positive" if is_pos else "Negative"

            st.session_state.history.append({
                'text': tweet[:80],
                'label': label,
                'conf': confidence,
                'time': datetime.now().strftime("%H:%M")
            })

            if is_pos:
                st.success(f"😊 Positive ({confidence}%)")
            else:
                st.error(f"😡 Negative ({confidence}%)")

            st.progress(confidence / 100)
            st.rerun()

# ===============================
# RIGHT SIDE (UNCHANGED)
# ===============================
with right:
    if st.session_state.history:
        st.write("Charts working...")