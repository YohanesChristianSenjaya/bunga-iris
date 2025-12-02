import joblib

def predict(data):
    # Memuat model yang sudah disimpan
    # Pastikan nama file sesuai dengan yang ada di folder Anda
    clf = joblib.load("knn_model.sav") 
    
    # Melakukan prediksi berdasarkan data input
    return clf.predict(data)