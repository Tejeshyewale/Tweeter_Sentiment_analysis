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
# Load Model
# ===============================
model = pickle.load(open(os.path.join(BASE_DIR, 'models', 'model.pkl'), 'rb'))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, 'models', 'vectorizer.pkl'), 'rb'))

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
# Custom CSS
# ===============================
st.markdown("""
<style>
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
    .metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 1.5rem; }
    .metric-box {
        background: #1a1a1a; border: 0.5px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 18px 20px;
    }
    .metric-val { font-size: 28px; font-weight: 700; color: #f0ede8; font-variant-numeric: tabular-nums; }
    .metric-lbl { font-size: 11px; color: #73726c; letter-spacing: 0.07em; text-transform: uppercase; margin-top: 4px; }

    /* Result cards */
    .result-pos {
        background: rgba(99,153,34,0.12); border: 1px solid rgba(99,153,34,0.35);
        border-radius: 14px; padding: 1.25rem 1.5rem; margin: 1rem 0;
    }
    .result-neg {
        background: rgba(226,75,74,0.12); border: 1px solid rgba(226,75,74,0.35);
        border-radius: 14px; padding: 1.25rem 1.5rem; margin: 1rem 0;
    }
    .result-title { font-size: 20px; font-weight: 700; margin: 0 0 4px; }
    .result-pos .result-title { color: #97C459; }
    .result-neg .result-title { color: #F09595; }
    .result-desc { font-size: 13px; color: #888; margin: 0; }

    /* History table */
    .hist-row {
        display: flex; align-items: center; gap: 12px;
        padding: 10px 0; border-bottom: 0.5px solid rgba(255,255,255,0.05);
        font-size: 13px;
    }
    .hist-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
    .badge {
        font-size: 11px; padding: 3px 10px; border-radius: 20px;
        font-weight: 500; flex-shrink: 0;
    }
    .badge-pos { background: rgba(99,153,34,0.2); color: #97C459; }
    .badge-neg { background: rgba(226,75,74,0.2); color: #F09595; }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background: transparent;
        border-bottom: 0.5px solid rgba(255,255,255,0.07);
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 13px; font-weight: 500; color: #73726c;
        background: transparent; border: none; padding: 8px 16px;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] { color: #1DA1F2 !important; }

    /* Input */
    .stTextArea textarea {
        background: #1a1a1a !important; border: 0.5px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important; color: #f0ede8 !important;
        font-family: 'Syne', sans-serif !important; font-size: 15px !important;
    }
    .stTextArea textarea:focus { border-color: #1DA1F2 !important; }

    /* Button */
    .stButton > button {
        background: #1DA1F2 !important; color: white !important;
        border: none !important; border-radius: 8px !important;
        font-family: 'Syne', sans-serif !important; font-weight: 500 !important;
        font-size: 14px !important; padding: 10px 24px !important;
        transition: opacity 0.15s !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }

    /* Chart background */
    .js-plotly-plot .plotly { background: transparent !important; }

    /* Section headers */
    .section-hdr {
        font-size: 11px; font-weight: 500; letter-spacing: 0.09em;
        text-transform: uppercase; color: #73726c; margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

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
# Metrics Row
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
# Main Layout: Input + Charts
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
        analyze_clicked = st.button("🔍 Analyze sentiment", use_container_width=True)
    with col_clr:
        if st.button("Clear", use_container_width=True):
            st.rerun()

    # ---- Prediction ----
    if analyze_clicked:
        if not tweet.strip():
            st.warning("Please enter some tweet text first.")
        else:
            processed = preprocess(tweet)
            vector = vectorizer.transform([processed])

            # Confidence score (if model supports predict_proba)
            try:
                proba = model.predict_proba(vector)[0]
                confidence = round(max(proba) * 100)
                predicted_class = int(model.classes_[proba.argmax()])
            except AttributeError:
                result = model.predict(vector)[0]
                predicted_class = int(result)
                confidence = 85  # fallback

            is_pos = predicted_class == 1
            label = "Positive" if is_pos else "Negative"

            # Save to history
            st.session_state.history.append({
                'text': tweet[:80] + ('…' if len(tweet) > 80 else ''),
                'label': label,
                'conf': confidence,
                'time': datetime.now().strftime("%H:%M")
            })

            # Display result
            if is_pos:
                st.markdown(f"""
                <div class="result-pos">
                  <div class="result-title">😊 Positive sentiment</div>
                  <div class="result-desc">The tweet expresses a positive tone · {confidence}% confidence</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-neg">
                  <div class="result-title">😡 Negative sentiment</div>
                  <div class="result-desc">The tweet expresses a negative tone · {confidence}% confidence</div>
                </div>""", unsafe_allow_html=True)

            # Confidence bar
            st.markdown('<div class="section-hdr" style="margin-top:12px">Confidence</div>', unsafe_allow_html=True)
            st.progress(confidence / 100)

            st.rerun()

    # ---- History ----
    if st.session_state.history:
        st.markdown('<br><div class="section-hdr">Recent analyses</div>', unsafe_allow_html=True)
        for h in reversed(st.session_state.history[-6:]):
            dot_color = "#639922" if h['label'] == 'Positive' else "#E24B4A"
            badge_cls = "badge-pos" if h['label'] == 'Positive' else "badge-neg"
            st.markdown(f"""
            <div class="hist-row">
              <div class="hist-dot" style="background:{dot_color}"></div>
              <div style="flex:1;color:#c8c5bf;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{h['text']}</div>
              <span class="badge {badge_cls}">{h['label']}</span>
              <span style="color:#555;font-size:11px;font-variant-numeric:tabular-nums">{h['conf']}%</span>
              <span style="color:#555;font-size:11px">{h['time']}</span>
            </div>""", unsafe_allow_html=True)

with right:
    if st.session_state.history:
        # ---- Donut chart ----
        st.markdown('<div class="section-hdr">Sentiment distribution</div>', unsafe_allow_html=True)
        fig_donut = go.Figure(go.Pie(
            labels=['Positive', 'Negative'],
            values=[pos_count, neg_count],
            hole=0.68,
            marker=dict(colors=['#639922', '#E24B4A'], line=dict(color='#0f0f0f', width=2)),
            textinfo='none',
            hovertemplate='%{label}: %{value} (%{percent})<extra></extra>'
        ))
        fig_donut.add_annotation(
            text=f"<b>{pos_rate}%</b><br><span style='font-size:10px'>Positive</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color='#f0ede8')
        )
        fig_donut.update_layout(
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5,
                        font=dict(color='#888', size=12)),
            margin=dict(t=10, b=30, l=10, r=10),
            height=220,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#888')
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        # ---- Confidence line chart ----
        if len(st.session_state.history) >= 2:
            st.markdown('<div class="section-hdr" style="margin-top:4px">Confidence over time</div>', unsafe_allow_html=True)
            last10 = st.session_state.history[-10:]
            labels_x = [f"#{i+1}" for i in range(len(last10))]
            colors_bar = ['#639922' if h['label'] == 'Positive' else '#E24B4A' for h in last10]

            fig_conf = go.Figure()
            fig_conf.add_trace(go.Scatter(
                x=labels_x,
                y=[h['conf'] for h in last10],
                mode='lines+markers',
                line=dict(color='#1DA1F2', width=2),
                marker=dict(color=colors_bar, size=8, line=dict(width=0)),
                fill='tozeroy',
                fillcolor='rgba(29,161,242,0.08)',
                hovertemplate='%{x}: %{y}%<extra></extra>'
            ))
            fig_conf.update_layout(
                height=180,
                margin=dict(t=5, b=20, l=30, r=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, tickfont=dict(color='#555', size=10)),
                yaxis=dict(range=[0, 105], showgrid=True, gridcolor='rgba(255,255,255,0.04)',
                           ticksuffix='%', tickfont=dict(color='#555', size=10)),
                showlegend=False
            )
            st.plotly_chart(fig_conf, use_container_width=True)

    else:
        st.markdown("""
        <div style="height:300px;display:flex;flex-direction:column;align-items:center;
                    justify-content:center;color:#444;text-align:center;gap:12px">
            <div style="font-size:36px">📊</div>
            <div style="font-size:14px">Charts appear after your first analysis</div>
        </div>""", unsafe_allow_html=True)