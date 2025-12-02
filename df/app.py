import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import os

# --- Config Halaman ---
st.set_page_config(page_title="Prediksi Rating TV Show", layout="centered")

# --- Load Data & Latih Model ---
@st.cache_resource
def train_model():
    # --- TEKNIK PATH ABSOLUT (SOLUSI ANTI GAGAL) ---
    # 1. Cari tahu di mana file 'app.py' ini tinggal
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Gabungkan alamat folder itu dengan nama file CSV
    # Pastikan nama filenya '10k_Poplar_Tv_Shows.csv' (Cek huruf 'Poplar' vs 'Popular')
    file_path = os.path.join(current_dir, '10k_Poplar_Tv_Shows.csv')
    
    try:
        # Load data menggunakan path lengkap
        df = pd.read_csv('10k_Poplar_Tv_Shows.csv')
        
        # Proses Data
        data = df[['overview', 'vote_average']].dropna()
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        X = tfidf.fit_transform(data['overview'])
        y = data['vote_average']
        
        # Latih Model
        model = LinearRegression()
        model.fit(X, y)
        return model, tfidf, None # Tidak ada error
        
    except FileNotFoundError:
        # Kembalikan pesan error detail untuk debugging
        return None, None, f"File tidak ditemukan di: {file_path}"
    except Exception as e:
        return None, None, str(e)

# Jalankan Training
model, tfidf, error_msg = train_model()

# --- Tampilan UI ---
st.title("⭐ AI Script Rater")

if model is None:
    st.error("❌ TERJADI ERROR SAAT LOAD DATA")
    st.code(error_msg) # Tampilkan path lengkap yang dicari komputer
    
    st.info("Debugging:")
    # Tampilkan file apa saja yang ada di folder ini
    current_dir = os.path.dirname(os.path.abspath(__file__))
    st.write(f"📂 Isi folder `{current_dir}` adalah:")
    st.write(os.listdir(current_dir))
    
    st.warning("""
    JIKA NAMA FILE CSV TIDAK MUNCUL DI LIST DI ATAS:
    1. Berarti file CSV belum ter-upload ke folder 'df' di GitHub.
    2. Atau nama filenya beda (misal: '10k_Popular...' vs '10k_Poplar...').
    """)
else:
    st.success("✅ Model Berhasil Dimuat!")
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