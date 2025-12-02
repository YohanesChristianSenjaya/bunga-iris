import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

# --- Config Halaman ---
st.set_page_config(page_title="Prediksi Rating TV Show", layout="centered")

# --- Load Data & Latih Model ---
@st.cache_resource
def train_model():
    try:
        # Pastikan nama file CSV sesuai dengan yang ada di GitHub
        df = pd.read_csv('10k_Popular_Tv_Shows.csv')
        # Hapus baris kosong agar tidak error saat training
        data = df[['overview', 'vote_average']].dropna()
        
        # Ubah teks ke angka
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        X = tfidf.fit_transform(data['overview'])
        y = data['vote_average']
        
        # Latih model
        model = LinearRegression()
        model.fit(X, y)
        return model, tfidf
    except Exception as e:
        return None, None

model, tfidf = train_model()

# --- Tampilan UI ---
st.title("⭐ AI Script Rater")

if model is None:
    st.error("Gagal memuat data/model.")
    st.info("Pastikan file '10k_Poplar_Tv_Shows.csv' ada di folder yang sama dengan app.py di GitHub.")
else:
    st.write("Tulis sinopsis (Bhs Inggris) untuk prediksi rating!")
    user_input = st.text_area("Sinopsis:", height=150)
    
    if st.button("Prediksi"):
        if user_input:
            vec = tfidf.transform([user_input])
            pred = model.predict(vec)[0]
            st.metric("Prediksi Rating", f"{pred:.2f}")