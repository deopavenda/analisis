import streamlit as st
import joblib
import re
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Inisialisasi
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
model = joblib.load("random_forest_model.pkl")

st.set_page_config(page_title="Analisis Sentimen Media Sosial", layout="centered")
st.title("Analisis Sentimen Media Sosial")
st.subheader("Prediksi Tren Pasar Menggunakan Random Forest")

# Fungsi preprocessing
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

# Session untuk riwayat prediksi
if "history" not in st.session_state:
    st.session_state.history = []

# Input pengguna
text_input = st.text_area("Masukkan teks media sosial di sini:")

if st.button("Prediksi Sentimen"):
    if text_input.strip():
        label, score = predict_sentiment(text_input)
        st.success(f"Hasil Prediksi: {label.upper()}")
        st.write(f"Skor Sentimen (VADER): {score:.4f}")

        # Simpan riwayat
        st.session_state.history.append({
            "Teks": text_input,
            "Label": label,
            "Skor": round(score, 4)
        })
    else:
        st.warning("Masukkan teks terlebih dahulu.")

# Tampilkan riwayat jika ada
if st.session_state.history:
    st.markdown("---")
    st.subheader("Riwayat Prediksi")

    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history)

    # Pie chart distribusi sentimen
    st.markdown("Distribusi Sentimen")
    fig, ax = plt.subplots()
    df_history["Label"].value_counts().plot.pie(
        autopct="%1.1f%%", startangle=90, ax=ax, colors=["#66bb6a", "#ef5350", "#ffee58"]
    )
    ax.set_ylabel("")
    st.pyplot(fig)

# Footer
st.markdown("<hr><center><sub>Dibuat oleh deopavenda • Aplikasi Streamlit</sub></center>", unsafe_allow_html=True)
