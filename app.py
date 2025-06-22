import streamlit as st
import joblib
import re
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Setup
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
model = joblib.load("random_forest_model.pkl")

st.set_page_config(page_title="Analisis Sentimen Media Sosial", layout="centered")
st.title("📊 Analisis Sentimen Media Sosial")
st.subheader("Prediksi Tren Pasar Menggunakan Random Forest")

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

# Session State untuk menyimpan riwayat prediksi
if "history" not in st.session_state:
    st.session_state.history = []

# Input pengguna
text_input = st.text_area("Masukkan teks media sosial di sini:")

if st.button("🔍 Prediksi Sentimen"):
    if text_input.strip():
        label, score = predict_sentiment(text_input)
        emoji = {"positive": "negative":, "neutral"
        st.success(f"Hasil Prediksi: **{label.upper()}** {emoji[label]}")
        st.write(f"Skor Sentimen (VADER): `{score:.4f}`")

        # Simpan ke history
        st.session_state.history.append({"Teks": text_input, "Label": label, "Skor": score})
    else:
        st.warning("⚠️ Harap masukkan teks terlebih dahulu.")

# Tampilkan Riwayat
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Riwayat Prediksi")

    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history)

    # Pie Chart
    st.markdown("### 📊 Distribusi Sentimen")
    fig, ax = plt.subplots()
    df_history["Label"].value_counts().plot.pie(autopct="%1.1f%%", startangle=90, ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

# Footer
st.markdown("<hr><center><sub>Developed by <b>deopavenda</b> • Streamlit App</sub></center>", unsafe_allow_html=True)
