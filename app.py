import streamlit as st
import joblib
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Unduh lexicon VADER
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# Load model
model = joblib.load("random_forest_model.pkl")

# Bersihkan teks
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

# Skor sentimen
def get_score(text):
    cleaned = clean_text(text)
    return sid.polarity_scores(cleaned)['compound']

# Prediksi sentimen
def predict_sentiment(text):
    score = get_score(text)
    return model.predict([[score]])[0]

# UI Streamlit
st.set_page_config(page_title="Analisis Sentimen", layout="centered")
st.title("📊 Analisis Sentimen Media Sosial")
st.subheader("Prediksi Tren Pasar Menggunakan Random Forest")

text_input = st.text_area("Masukkan teks media sosial di sini:")
if st.button("Prediksi Sentimen"):
    if not text_input.strip():
        st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")
    else:
        result = predict_sentiment(text_input)
        st.success(f"📝 Prediksi Sentimen: **{result.upper()}**")
        st.write(f"Skor Sentimen (VADER): `{get_score(text_input):.4f}`")
