# app.py

import streamlit as st
import requests
import time
from datetime import datetime

API_URL = "http://127.0.0.1:8000/predict"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fake News Detection (Real-Time)",
    page_icon="📰",
    layout="centered"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 Fake News Detector")
st.sidebar.markdown(
    """
🔄 **Real-time prediction enabled**

Predictions update automatically as you type.
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ Predictions are probabilistic, not fact checks.")

# ---------------- SESSION STATE ----------------
if "last_text" not in st.session_state:
    st.session_state.last_text = ""

if "last_prediction_time" not in st.session_state:
    st.session_state.last_prediction_time = 0

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- MAIN ----------------
st.title("📰 Real-Time Fake News Detection")
st.write("Start typing news text — prediction updates automatically.")

# -------- EXTRA QUESTIONS --------
source = st.selectbox(
    "📍 Where did you see this news?",
    ["Unknown", "News Website", "Social Media", "WhatsApp / Telegram", "TV / YouTube"]
)

content_type = st.radio(
    "🗂️ Content Type",
    ["Headline", "Full Article", "Social Media Post"],
    horizontal=True
)

user_trust = st.selectbox(
    "🤔 Do you trust this source?",
    ["Not sure", "Yes", "No"]
)

# -------- TEXT INPUT --------
text_input = st.text_area(
    "📝 News Text",
    height=160,
    placeholder="Government announces miracle cure overnight..."
)

# ---------------- REAL-TIME LOGIC ----------------
MIN_WORDS = 5
DEBOUNCE_SECONDS = 1.2

current_time = time.time()
word_count = len(text_input.split())

should_predict = (
    text_input.strip() != ""
    and word_count >= MIN_WORDS
    and text_input != st.session_state.last_text
    and current_time - st.session_state.last_prediction_time > DEBOUNCE_SECONDS
)

if should_predict:
    with st.spinner("Analyzing..."):
        try:
            response = requests.post(
                API_URL,
                json={"text": text_input},
                timeout=5
            )

            if response.status_code == 200:
                result = response.json()
                label = result["label"]
                confidence = result["confidence"]

                st.session_state.last_text = text_input
                st.session_state.last_prediction_time = current_time

                # -------- DISPLAY RESULT --------
                st.markdown("## 📊 Live Prediction")

                if label == "FAKE":
                    st.error("🚨 FAKE NEWS")
                elif label == "REAL":
                    st.success("✅ REAL NEWS")
                else:
                    st.info("ℹ️ UNKNOWN")

                st.write(f"**Confidence:** {confidence:.2f}")
                st.progress(min(confidence, 1.0))

                st.caption(
                    "Prediction updates automatically when you pause typing."
                )

                # -------- CONTEXT --------
                st.markdown("### 🧾 Context")
                st.write(f"- **Source:** {source}")
                st.write(f"- **Content Type:** {content_type}")
                st.write(f"- **User Trust:** {user_trust}")

                # -------- SAVE HISTORY --------
                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "label": label,
                    "confidence": confidence,
                    "text": text_input[:60] + "..."
                })

            else:
                st.warning("API error. Check backend.")

        except requests.exceptions.RequestException:
            st.error("Backend not reachable.")

elif text_input.strip() and word_count < MIN_WORDS:
    st.info(f"Type at least {MIN_WORDS} words for live prediction.")

# ---------------- HISTORY ----------------
if st.session_state.history:
    st.markdown("---")
    st.markdown("## 🕒 Recent Live Predictions")

    for item in reversed(st.session_state.history[-5:]):
        st.markdown(
            f"- **{item['time']}** | "
            f"{item['label']} ({item['confidence']:.2f}) — "
            f"{item['text']}"
        )
