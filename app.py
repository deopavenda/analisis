import streamlit as st
import joblib
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Setup
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
model = joblib.load("random_forest_model.pkl")

# Styling tambahan
st.set_page_config(page_title="Sentimen Sosial Media", layout="centered")
st.markdown(
    "<h1 style='text-align: center;'>📊 Analisis Sentimen Media Sosial</h1>"
    "<h4 style='text-align: center; color: gray;'>Prediksi Tren Pasar Menggunakan Random Forest</h4><br>",
    unsafe_allow_html=True
)

# Fungsi bantu
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

def get_score(text):
    cleaned = clean_text(text)
    return sid.polarity_scores(cleaned)['compound']

def predict_sentiment(text):
    score = get_score(text)
    return model.predict([[score]])[0], score

# Input box
st.write("💬 Masukkan teks media sosial di bawah ini:")
text_input = st.text_area("", placeholder="Contoh: Saham Tesla sangat menjanjikan hari ini 🚀")

# Button prediksi
if st.button("🔍 Prediksi Sentimen"):
    if not text_input.strip():
        st.warning("⚠️ Masukkan teks terlebih dahulu.")
    else:
        result, score = predict_sentiment(text_input)
        
        # Emoji hasil
        emoji = {
            "positive": "😊",
            "negative": "😠",
            "neutral": "😐"
        }

        # Warna hasil
        color = {
            "positive": "green",
            "negative": "red",
            "neutral": "gray"
        }

        st.markdown(f"<h3 style='color:{color[result]};'>Hasil: {result.upper()} {emoji[result]}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p><b>Skor Sentimen (VADER):</b> {score:.4f}</p>", unsafe_allow_html=True)

# Footer
st.markdown("<hr><p style='text-align:center; font-size:12px;'>Dibuat oleh <b>deopavenda</b> • Streamlit + Random Forest + VADER</p>", unsafe_allow_html=True)
