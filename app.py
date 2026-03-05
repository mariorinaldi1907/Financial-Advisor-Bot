# app.py
# Financial Advisor Bot (SPY Prototype) — Streamlit Dashboard (Polished UI)
# -----------------------------------------------------------------------
# Requirements (inside your .venv):
#   pip install streamlit pandas numpy joblib scikit-learn yfinance plotly
#
# Run:
#   python -m streamlit run app.py

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# -----------------------------
# Page / Theme
# -----------------------------
st.set_page_config(page_title="Financial Advisor Bot", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
      .big-title {font-size: 30px; font-weight: 800; margin-bottom: 0px;}
      .sub-title {font-size: 14px; opacity: 0.85; margin-top: 4px;}

      .card {
        padding: 14px;
        border-radius: 14px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
      }

      .good {
        padding: 12px 14px;
        border-radius: 14px;
        background: rgba(0, 200, 0, 0.10);
        border: 1px solid rgba(0, 200, 0, 0.25);
      }

      .bad {
        padding: 12px 14px;
        border-radius: 14px;
        background: rgba(255, 0, 0, 0.08);
        border: 1px solid rgba(255, 0, 0, 0.22);
      }

      .neutral {
        padding: 12px 14px;
        border-radius: 14px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
      }

      /* Compact KPI cards */
      .kpi-wrap {display: flex; gap: 10px; margin-top: 6px; margin-bottom: 10px;}
      .kpi {
        flex: 1;
        padding: 10px 12px;
        border-radius: 14px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
      }
      .kpi-label {font-size: 12px; opacity: 0.8; margin-bottom: 4px;}
      .kpi-value {font-size: 18px; font-weight: 700; line-height: 1.2;}
      .kpi-value.small {font-size: 16px;}
      .small-note {font-size: 12px; opacity: 0.8;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='big-title'>📈 Financial Advisor Bot (SPY Prototype)</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>10-day direction forecast using technical indicators + Random Forest, with explainable recommendations.</div>",
    unsafe_allow_html=True
)
st.write("")

# -----------------------------
# Load Model + Data
# -----------------------------
MODEL_PATH = os.path.join("models", "rf_model_spy.pkl")
DATA_PATH = os.path.join("data", "spy_processed.csv")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: {MODEL_PATH}. Make sure you saved it from the notebook.")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found at: {DATA_PATH}. Make sure you saved spy_processed.csv from the notebook.")
    st.stop()

model = load_model()
df = load_data()

# -----------------------------
# Expected Columns / Features
# -----------------------------
feature_cols = [
    "SMA_10", "SMA_30", "SMA_50", "SMA_100",
    "EMA_12", "EMA_26",
    "MACD", "MACD_signal",
    "RSI_14",
    "Volatility_20", "Volatility_50",
    "Mom_5", "Mom_10", "Mom_20",
    "VROC_5",
    "Volume"
]

missing_feats = [c for c in feature_cols if c not in df.columns]
if missing_feats:
    st.error(
        "Your processed CSV is missing feature columns required by the model:\n\n"
        + ", ".join(missing_feats)
        + "\n\nFix: ensure your notebook saves the final modelling dataframe with these columns."
    )
    st.stop()

# Drop NaNs created by rolling indicators
df = df.dropna(subset=feature_cols + (["Close"] if "Close" in df.columns else []), how="any")

# Ensure index is sorted
df = df.sort_index()

available_dates = df.index

# -----------------------------
# Sidebar Inputs (Cleaner Date UI)
# -----------------------------
st.sidebar.header("User Inputs")

investment_amount = st.sidebar.slider(
    "Investment Amount ($)",
    min_value=100,
    max_value=50000,
    value=5000,
    step=100
)

risk_level = st.sidebar.selectbox("Risk Level", ["Low", "Moderate", "High"])

# Date picker (looks much cleaner than dropdown)
min_d = available_dates.min().date()
max_d = available_dates.max().date()

picked_date = st.sidebar.date_input(
    "Select Date",
    value=max_d,
    min_value=min_d,
    max_value=max_d
)

st.sidebar.markdown("---")
st.sidebar.caption("SPY-only prototype (dates are constrained to available trading days).")
st.sidebar.markdown("<div class='small-note'>Tip: For report screenshots, capture one UP case and one DOWN case.</div>", unsafe_allow_html=True)

# Convert picked_date -> Timestamp at midnight
picked_ts = pd.Timestamp(picked_date)

# Snap to nearest available trading date if weekend/holiday
# (choose closest by absolute time difference)
nearest_idx = int(np.argmin(np.abs((available_dates - picked_ts).to_numpy())))
selected_date = available_dates[nearest_idx]

run = st.sidebar.button("Get Recommendation")

# -----------------------------
# Helper: Explanation Generator
# -----------------------------
def build_explanations(row: pd.Series):
    reasons = []

    if row["SMA_30"] > row["SMA_100"]:
        reasons.append("Short-term trend is stronger than long-term trend (SMA 30 > SMA 100).")
    else:
        reasons.append("Short-term trend is weaker than long-term trend (SMA 30 ≤ SMA 100).")

    if row["MACD"] > 0:
        reasons.append("MACD is above zero, suggesting bullish momentum.")
    else:
        reasons.append("MACD is below zero, suggesting weak or bearish momentum.")

    if row["RSI_14"] > 70:
        reasons.append("RSI suggests overbought conditions (potential pullback risk).")
    elif row["RSI_14"] < 30:
        reasons.append("RSI suggests oversold conditions (potential rebound signal).")
    else:
        reasons.append("RSI is in a neutral range.")

    if row["Mom_10"] > 0:
        reasons.append("Recent 10-day momentum is positive.")
    else:
        reasons.append("Recent 10-day momentum is negative.")

    if row["Volatility_20"] > row["Volatility_50"]:
        reasons.append("Short-term volatility is elevated relative to the longer-term window.")
    else:
        reasons.append("Volatility appears stable relative to the longer-term window.")

    return reasons

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1.25, 1.0], gap="large")

# -----------------------------
# Chart (Left)
# -----------------------------
with left:
    st.subheader("📉 Market Visualisation")

    end_date = selected_date
    recent_data = df.loc[:end_date].tail(140).copy()

    fig = go.Figure()

    if "Close" in recent_data.columns:
        fig.add_trace(go.Scatter(
            x=recent_data.index, y=recent_data["Close"],
            mode="lines", name="Close"
        ))

    if "SMA_30" in recent_data.columns:
        fig.add_trace(go.Scatter(
            x=recent_data.index, y=recent_data["SMA_30"],
            mode="lines", name="SMA 30"
        ))

    if "SMA_100" in recent_data.columns:
        fig.add_trace(go.Scatter(
            x=recent_data.index, y=recent_data["SMA_100"],
            mode="lines", name="SMA 100"
        ))

    fig.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Date",
        yaxis_title="Price"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Selected trading date used by the model: {selected_date.date()}")

# -----------------------------
# Prediction & Advice (Right)
# -----------------------------
with right:
    st.subheader("📊 Prediction & Advice")

    if not run:
        st.info("Pick inputs on the left, then click **Get Recommendation**.")
    else:
        row = df.loc[selected_date]
        X_input = pd.DataFrame([row[feature_cols]], columns=feature_cols)

        # Predict + confidence
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0]
            prediction = int(np.argmax(proba))
            confidence = float(np.max(proba)) * 100
        else:
            prediction = int(model.predict(X_input)[0])
            confidence = None

        direction = "UP" if prediction == 1 else "DOWN"
        action = "Increase Exposure" if direction == "UP" else "Reduce / Hold"

        # Compact KPI row (smaller than st.metric)
        conf_text = f"{confidence:.1f}%" if confidence is not None else "N/A"

        st.markdown(
            f"""
            <div class="kpi-wrap">
              <div class="kpi">
                <div class="kpi-label">Predicted Direction</div>
                <div class="kpi-value">{direction}</div>
              </div>
              <div class="kpi">
                <div class="kpi-label">Suggested Action</div>
                <div class="kpi-value small">{action}</div>
              </div>
              <div class="kpi">
                <div class="kpi-label">Risk Level</div>
                <div class="kpi-value small">{risk_level}</div>
              </div>
              <div class="kpi">
                <div class="kpi-label">Model Confidence</div>
                <div class="kpi-value small">{conf_text}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Recommendation (risk-aware)
        if direction == "UP":
            if risk_level == "High":
                alloc_text = "a higher allocation may be considered (risk-tolerant approach)."
            elif risk_level == "Low":
                alloc_text = "a cautious increase or partial allocation may be appropriate."
            else:
                alloc_text = "a balanced allocation is recommended."
            recommendation = (
                f"The model predicts SPY may rise over the next 10 trading days. "
                f"For an investment of ${investment_amount:,}, {alloc_text}"
            )
            st.markdown(f"<div class='good'><b>💡 Recommendation</b><br>{recommendation}</div>", unsafe_allow_html=True)
        else:
            recommendation = (
                f"The model predicts SPY may not rise over the next 10 trading days. "
                f"For an investment of ${investment_amount:,}, a reduced allocation or defensive positioning may be appropriate."
            )
            st.markdown(f"<div class='bad'><b>💡 Recommendation</b><br>{recommendation}</div>", unsafe_allow_html=True)

        st.write("")

        # Explanation
        reasons = build_explanations(row)
        st.markdown("<div class='neutral'><b>🔎 Explanation (Indicator-Based)</b><ul>", unsafe_allow_html=True)
        for r in reasons:
            st.markdown(f"<li>{r}</li>", unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)

        st.write("")
        st.caption("Note: This prototype is an educational advisor system and does not constitute financial advice.")