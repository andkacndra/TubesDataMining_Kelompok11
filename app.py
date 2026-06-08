import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. JUDUL UTAMA APLIKASI
st.set_page_config(page_title="E-commerce Loyalty Program & Membership Predictor", layout="centered")
st.title("E-commerce Loyalty Program & Membership Predictor")

# 2. KETERANGAN DESKRIPSI
st.write("Aplikasi ini memprediksi tingkat keanggotaan/loyalitas pelanggan baru berdasarkan model Logistic Regression kelompok kami.")

# Load model dan scaler .pkl
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
    st.error("File model_logreg.pkl atau scaler.pkl tidak ditemukan di folder ini!")

# --- FORM INPUT DATA PELANGGAN BARU ---
st.header("Analis Data Pelanggan Baru")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Jenis Kelamin (Gender)", ["Female", "Male"])
    age = st.number_input("Umur (Age)", min_value=1, max_value=100, value=30)
    
    # Input dari user berupa tingkat kepuasan (sesuai revisi ketua)
    satisfaction = st.selectbox("Tingkat Kepuasan Pelanggan (Satisfaction Level)", ["Unsatisfied", "Neutral", "Satisfied"])
    
    total_spend = st.number_input("Total Pengeluaran (Total Spend $)", min_value=0.0, value=500.0)

with col2:
    items = st.number_input("Jumlah Barang Dibeli (Items Purchased)", min_value=0, value=10)
    rating = st.slider("Rata-rata Rating (Average Rating)", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
    discount = st.selectbox("Mendapatkan Diskon? (Discount Applied)", ["No", "Yes"])
    days_since = st.number_input("Hari Sejak Transaksi Terakhir (Days Since Last Purchase)", min_value=0, value=15)

# Pilihan kota
city = st.selectbox("Kota Tempat Tinggal (City)", ["Chicago", "Houston", "Los Angeles", "Miami", "New York", "San Francisco"])

# --- PROSES TOMBOL EKSEKUSI ---
if st.button("Analisis Tingkat Keanggotaan"):
    
    # Mapping Data Sesuai Aturan Encoding
    gender_encoded = 0 if gender == "Female" else 1
    discount_encoded = 1 if discount == "Yes" else 0
    
    # Konversi teks kepuasan menjadi angka untuk dikirim ke posisi latih awal model
    if satisfaction == "Unsatisfied": satisfaction_encoded = 0
    elif satisfaction == "Neutral": satisfaction_encoded = 1
    else: satisfaction_encoded = 2
    
    # Inisialisasi One-Hot Encoding Kota
    cities = ["Chicago", "Houston", "Los Angeles", "Miami", "New York", "San Francisco"]
    # FIXED: Mengubah 'for f in cities' menjadi 'for c in cities' agar variabel 'c' terbaca sempurna
    city_dict = {f"City_{c}": 0 for c in cities}
    city_dict[f"City_{city}"] = 1
    
    # Susunan asli kolom Vian agar scaler.transform tidak komplain nama kolom berbeda
    input_data = {
        'Gender': gender_encoded,
        'Age': age,
        'Membership Type': satisfaction_encoded,  # Mengisi posisi kolom latih dengan data input kepuasan
        'Total Spend': total_spend,
        'Items Purchased': items,
        'Average Rating': rating,
        'Discount Applied': discount_encoded,
        'Days Since Last Purchase': days_since,
        **city_dict
    }
    
    # Ubah ke DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Standardisasi data menggunakan scaler bawaan
    df_input_scaled = scaler.transform(df_input)
    
    # Prediksi menggunakan model Logistic Regression
    prediction = model.predict(df_input_scaled)[0]
    
    # OUTPUT HASIL PREDIKSI (BRONZE, SILVER, GOLD)
    st.subheader("Hasil Analisis Model:")
    if prediction == 0:
        st.error("Hasil Prediksi: **BRONZE MEMBER**")
    elif prediction == 1:
        st.warning("Hasil Prediksi: **SILVER MEMBER**")
    else:
        st.success("Hasil Prediksi: **GOLD MEMBER**")