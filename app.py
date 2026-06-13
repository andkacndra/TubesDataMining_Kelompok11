import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Configuration & Title
st.set_page_config(page_title="E-commerce Loyalty Program & Membership Predictor", layout="centered")
st.title("📊 E-commerce Loyalty Program & Membership Predictor")
st.write("Aplikasi ini memprediksi tingkat keanggotaan/loyalitas pelanggan baru berdasarkan model Logistic Regression kelompok kami.")

# Load models safely
@st.cache_resource
def load_models():
    with open('model_logreg.pkl', 'rb') as f_model:
        model = pickle.load(f_model)
    with open('scaler.pkl', 'rb') as f_scaler:
        scaler = pickle.load(f_scaler)
    return model, scaler

try:
    model, scaler = load_models()
except FileNotFoundError:
    st.error("File model_logreg.pkl atau scaler.pkl tidak ditemukan!")

# --- FORM INPUT DATA PELANGGAN BARU ---
st.header("Analis Data Pelanggan Baru")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Jenis Kelamin (Gender)", ["Female", "Male"])
    age = st.number_input("Umur (Age)", min_value=1, max_value=100, value=30)
    total_spend = st.number_input("Total Pengeluaran (Total Spend $)", min_value=0.0, value=500.0)
    items = st.number_input("Jumlah Barang Dibeli (Items Purchased)", min_value=0, value=10)

with col2:
    rating = st.slider("Rata-rata Rating (Average Rating)", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
    discount = st.selectbox("Mendapatkan Diskon? (Discount Applied)", ["No", "Yes"])
    days_since = st.number_input("Hari Sejak Transaksi Terakhir (Days Since Last Purchase)", min_value=0, value=15)
    
    # Input Satisfaction Level sesuai instruksi visual Ketua Kelompok
    satisfaction = st.selectbox("Tingkat Kepuasan Pelanggan (Satisfaction Level)", ["Unsatisfied", "Neutral", "Satisfied"])

# Pilihan Kota untuk One-Hot Encoding
city = st.selectbox("Kota Tempat Tinggal (City)", ["Chicago", "Houston", "Los Angeles", "Miami", "New York", "San Francisco"])

# --- PROSES TOMBOL ANALISIS ---
if st.button("🔮 Analisis Tingkat Keanggotaan"):
    
    # Mapping Data secara manual dan mutlak
    gender_encoded = 0 if gender == "Female" else 1
    discount_encoded = 1 if discount == "Yes" else 0
    
    if satisfaction == "Unsatisfied": satisfaction_encoded = 0
    elif satisfaction == "Neutral": satisfaction_encoded = 1
    else: satisfaction_encoded = 2
    
    city_chicago = 1 if city == "Chicago" else 0
    city_houston = 1 if city == "Houston" else 0
    city_los_angeles = 1 if city == "Los Angeles" else 0
    city_miami = 1 if city == "Miami" else 0
    city_new_york = 1 if city == "New York" else 0
    city_san_francisco = 1 if city == "San Francisco" else 0
    
    # Menyusun list data mentah sesuai urutan 14 kolom scaler Vian secara eksak
    raw_features = [
        gender_encoded, age, total_spend, items, rating, 
        discount_encoded, days_since, satisfaction_encoded, 
        city_chicago, city_houston, city_los_angeles, city_miami, 
        city_new_york, city_san_francisco
    ]
    
    # Ubah menjadi matriks 2D NumPy Array untuk mematikan validasi kolom sklearn
    df_input_array = np.array([raw_features])
    
    # TAMPILKAN DEBUG DATA (Untuk intip angka sebelum masuk scaler)
    st.info(f"💡 **Data mentah yang dikirim ke model:** {raw_features}")
    
    # Standardisasi data lewat array murni
    df_input_scaled = scaler.transform(df_input_array)
    
    # Cek peluang probabilitas tiap kelas (Bronze, Silver, Gold)
    probabilities = model.predict_proba(df_input_scaled)[0]
    st.info(f"📊 **Peluang Probabilitas Kelas [Bronze, Silver, Gold]:** {np.round(probabilities, 4)}")
    
    # Prediksi kelas target (1 = Bronze, 2 = Silver, 3 = Gold)
    prediction = model.predict(df_input_scaled)[0]
    
    # Output Hasil Berdasarkan Label Target Vian
    st.subheader("🎯 Hasil Analisis Model:")
    if prediction == 1:
        st.error("🥉 Hasil Prediksi: **BRONZE MEMBER**")
    elif prediction == 2:
        st.warning("🥈 Hasil Prediksi: **SILVER MEMBER**")
    elif prediction == 3:
        st.success("🥇 Hasil Prediksi: **GOLD MEMBER**")