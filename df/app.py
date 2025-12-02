import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- Load Data & Model (Bisa di-cache agar cepat) ---
@st.cache_data
def load_data():
    df = pd.read_csv('10k_Poplar_Tv_Shows.csv')
    df = df[['name', 'overview']].dropna().reset_index(drop=True)
    return df

df = load_data()

# Menyiapkan model TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['name']).drop_duplicates()

def get_recommendations(title):
    if title not in indices:
        return []
    idx = indices[title]
    # Handle jika ada duplikat judul, ambil yang pertama
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]
        
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['name'].iloc[movie_indices].tolist()

# --- Tampilan Streamlit ---
st.title("Sistem Rekomendasi Acara TV")
st.write("Pilih acara TV favoritmu, dan kami akan merekomendasikan yang mirip!")

selected_movie = st.selectbox("Pilih Acara TV:", df['name'].values)

if st.button("Rekomendasikan"):
    recommendations = get_recommendations(selected_movie)
    if recommendations:
        st.write("Kami merekomendasikan:")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    else:
        st.write("Maaf, tidak ada rekomendasi yang ditemukan.")