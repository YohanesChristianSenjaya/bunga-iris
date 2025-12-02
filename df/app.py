import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Set konfigurasi halaman
st.set_page_config(page_title="Rekomendasi TV Show", layout="wide")

# Fungsi load data dengan caching agar cepat
@st.cache_data
def load_data():
    # Pastikan file csv ada di satu folder dengan app.py
    df = pd.read_csv('10k_Poplar_Tv_Shows.csv')
    df = df[['name', 'overview', 'poster_path', 'vote_average']].dropna().reset_index(drop=True)
    return df

# Fungsi untuk menyiapkan model (dijalankan sekali saat app mulai)
@st.cache_resource
def prepare_model(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['name']).drop_duplicates()
    return cosine_sim, indices

# --- Main App ---
st.title("🎬 TV Show Recommender System")
st.write("Bingung mau nonton apa? Pilih acara favoritmu, kami carikan yang mirip!")

# Load data & model
try:
    df = load_data()
    cosine_sim, indices = prepare_model(df)
    
    # Dropdown untuk memilih film
    selected_show = st.selectbox("Pilih Acara TV yang kamu suka:", df['name'].values)

    if st.button("Cari Rekomendasi"):
        with st.spinner('Mencari acara yang mirip...'):
            # Logika rekomendasi
            idx = indices[selected_show]
            if isinstance(idx, pd.Series):
                idx = idx.iloc[0]
            
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:6] # Ambil 5 rekomendasi teratas
            
            movie_indices = [i[0] for i in sim_scores]
            recommendations = df.iloc[movie_indices]

            # Tampilkan hasil dalam kolom
            st.success(f"Karena kamu suka **{selected_show}**, mungkin kamu juga suka:")
            
            cols = st.columns(5)
            for i, col in enumerate(cols):
                with col:
                    row = recommendations.iloc[i]
                    st.write(f"**{row['name']}**")
                    st.caption(f"⭐ Rating: {row['vote_average']}")
                    # Tampilkan poster jika link valid (opsional, tergantung data)
                    if row['poster_path']:
                        full_path = "https://image.tmdb.org/t/p/w500" + row['poster_path']
                        st.image(full_path)
                    with st.expander("Sinopsis"):
                        st.write(row['overview'][:100] + "...")

except FileNotFoundError:
    st.error("File '10k_Poplar_Tv_Shows.csv' tidak ditemukan. Pastikan file ada di folder yang sama!")