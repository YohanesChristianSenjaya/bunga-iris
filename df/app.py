import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# --- Config ---
st.set_page_config(page_title="Prediksi Rating TV Show", layout="centered")

# --- Load Data & Train Model (Cached) ---
@st.cache_resource
def train_model():
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

# Load model saat aplikasi dibuka
try:
    model, tfidf = train_model()

    # --- UI Streamlit ---
    st.title("⭐ TV Show Rating Predictor")
    st.write("Masukan ide cerita (sinopsis) kamu dalam bahasa Inggris, dan AI akan memprediksi berapa rating yang mungkin didapatkan!")

    # Input User
    user_input = st.text_area("Tulis Sinopsis di sini:", height=150, 
                              placeholder="Example: A young wizard goes to a magic school and fights a dark lord...")

    if st.button("Prediksi Rating"):
        if user_input:
            # Proses input
            input_vec = tfidf.transform([user_input])
            prediction = model.predict(input_vec)[0]
            
            # Tampilkan hasil
            st.subheader("Hasil Prediksi:")
            st.metric(label="Perkiraan Rating (0-10)", value=f"{prediction:.2f}")
            
            # Feedback visual
            if prediction > 7.5:
                st.success("Wah, sepertinya ide ceritamu sangat menarik! 🤩")
            elif prediction > 5.0:
                st.warning("Cerita yang lumayan, standar acara TV pada umumnya. 🙂")
            else:
                st.error("Hmm, mungkin perlu perbaikan alur cerita agar penonton suka. 😅")
        else:
            st.warning("Mohon isi sinopsis terlebih dahulu.")

except FileNotFoundError:
    st.error("File dataset tidak ditemukan.")