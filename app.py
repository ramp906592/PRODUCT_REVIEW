import streamlit as st
import joblib
import re
import numpy as np

# Load model
model = joblib.load("logistic_regression_model.pkl")
tfidf = joblib.load("tfidf_vectorizer (1).pkl")

# Page Config
st.set_page_config(
    page_title="Amazon Review Analyzer",
    page_icon="🛒",
    layout="wide"
)

# ----------------------- GLOBAL CSS FIX ------------------------
st.markdown("""
<style>

/* Main background */
body {
    background-color: #0e0f11 !important;
}

/* Remove top white bar */
.block-container {
    padding-top: 1rem !important;
}

/* Textarea styling */
textarea {
    background: #1c1e21 !important;
    color: white !important;
    border: 2px solid #3a3b3d !important;
    border-radius: 12px !important;
}

/* Title styling */
.title {
    text-align: center;
    font-size: 40px !important;
    font-weight: bold;
    color: white;
}

/* Result card styling */
.result-card {
    padding: 25px;
    background: #fffbe6;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.3);
    margin-top: 20px;
    color: #333 !important;
}

/* FIXED FOOTER */
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    padding: 12px;
    background: #0e0f11;
    text-align: center;
    color: #999999;
    font-size: 15px;
    border-top: 1px solid #333;
}

</style>
""", unsafe_allow_html=True)

# ----------------------- TEXT CLEANING ------------------------
def preprocess(text):
    text = text.lower().strip()
    text = re.sub(r'[^\x00-\x7F]+',' ', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# ----------------------- UI TITLE ------------------------
st.markdown("<h1 class='title'>🛒 Amazon Product Review Analyzer</h1>", unsafe_allow_html=True)
st.write("<p style='text-align:center; color:#ccc;'>Enter a product review and get instant AI sentiment analysis.</p>", unsafe_allow_html=True)

# Textbox
review = st.text_area("Write Review:", height=140, placeholder="Example: Amazing product, great quality!")

# ----------------------- PREDICT BUTTON ------------------------
if st.button("Analyze Sentiment", use_container_width=True):
    if review.strip() == "":
        st.warning("⚠ Please enter a review.")
    else:
        clean_text = preprocess(review)
        vector = tfidf.transform([clean_text])

        pred = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]

        sentiment = "😀 Positive Review" if pred == 1 else "😡 Negative Review"
        confidence = round(max(prob) * 100, 2)

        st.markdown(f"""
        <div class='result-card'>
            <h2>{sentiment}</h2>
            <p><b>Confidence:</b> {confidence}%</p>
        </div>
        """, unsafe_allow_html=True)

# ----------------------- FIXED FOOTER ------------------------
st.markdown("""
<div class='footer'>
    Developed by <b>Ram Prakash Jha</b>
</div>
""", unsafe_allow_html=True)
