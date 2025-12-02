# --- BAGIAN 1: INSTALL PAKSA (WAJIB DI PALING ATAS) ---
import os
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Cek apakah scikit-learn sudah ada, jika belum, install sekarang juga
try:
    import sklearn
except ImportError:
    install("scikit-learn")
    import sklearn

# --- BAGIAN 2: KODE APLIKASI ANDA ---
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

# --- Config Halaman ---
st.set_page_config(page_title="Prediksi Rating TV Show", layout="centered")

# --- Load Data & Latih Model ---
@st.cache_resource
def train_model():
    # Pastikan file CSV ada di folder yang sama
    try:
        df = pd.read_csv('10k_Poplar_Tv_Shows.csv')
        data = df[['overview', 'vote_average']].dropna()
        
        # Vectorizer
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        X = tfidf.fit_transform(data['overview'])
        y = data['vote_average']
        
        # Model
        model = LinearRegression()
        model.fit(X, y)
        return model, tfidf
    except FileNotFoundError:
        return None, None

# Jalankan training
model, tfidf = train_model()

# --- Tampilan Streamlit ---
st.title("⭐ AI Script Rater")

if model is None:
    st.error("❌ File '10k_Poplar_Tv_Shows.csv' tidak ditemukan!")
    st.warning("Pastikan file CSV sudah di-upload ke GitHub di folder yang sama dengan app.py")
else:
    st.markdown("Tulis sinopsis cerita (Bhs Inggris), AI akan memprediksi ratingnya!")
    
    user_input = st.text_area("Sinopsis:", height=150, 
                              placeholder="Example: A magical school...")

    if st.button("Prediksi"):
        if user_input:
            input_vec = tfidf.transform([user_input])
            prediction = model.predict(input_vec)[0]
            final_score = max(0, min(10, prediction))
            
            st.metric("Prediksi Rating", f"{final_score:.2f}")
        else:
            st.warning("Isi sinopsis dulu.")