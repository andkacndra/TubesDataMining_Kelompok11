import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set konfigurasi halaman dashboard
st.set_page_config(page_title="E-commerce Customer Satisfaction Predictor", layout="centered")
st.title("E-commerce Customer Satisfaction Predictor")
st.write("Aplikasi ini memprediksi tingkat kepuasan pelanggan berdasarkan model Logistic Regression kelompok kami.")

# Fungsi untuk memuat model dan scaler .pkl
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
    membership = st.selectbox("Tipe Keanggotaan (Membership Type)", ["Bronze", "Silver", "Gold"])
    total_spend = st.number_input("Total Pengeluaran (Total Spend $)", min_value=0.0, value=500.0)

with col2:
    items = st.number_input("Jumlah Barang Dibeli (Items Purchased)", min_value=0, value=10)
    rating = st.slider("Rata-rata Rating (Average Rating)", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
    discount = st.selectbox("Mendapatkan Diskon? (Discount Applied)", ["No", "Yes"])
    days_since = st.number_input("Hari Sejak Transaksi Terakhir (Days Since Last Purchase)", min_value=0, value=15)

# Pilihan kota sesuai One-Hot Encoding Okta
city = st.selectbox("Kota Tempat Tinggal (City)", ["Chicago", "Houston", "Los Angeles", "Miami", "New York", "San Francisco"])

# --- PROSES TOMBOL PREDIKSI ---
if st.button("🔮 Prediksi Kepuasan Pelanggan"):
    
    # Mapping Data Sesuai Aturan Encoding Okta
    gender_encoded = 0 if gender == "Female" else 1
    discount_encoded = 1 if discount == "Yes" else 0
    
    if membership == "Bronze": membership_encoded = 1
    elif membership == "Silver": membership_encoded = 2
    else: membership_encoded = 3
    
    # Inisialisasi One-Hot Encoding Kota menjadi 0 semua
    cities = ["Chicago", "Houston", "Los Angeles", "Miami", "New York", "San Francisco"]
    city_dict = {f"City_{c}": 0 for c in cities}
    city_dict[f"City_{city}"] = 1
    
    # Satukan data sesuai urutan kolom dataset asli Vian
    input_data = {
        'Gender': gender_encoded,
        'Age': age,
        'Membership Type': membership_encoded,
        'Total Spend': total_spend,
        'Items Purchased': items,
        'Average Rating': rating,
        'Discount Applied': discount_encoded,
        'Days Since Last Purchase': days_since,
        **city_dict
    }
    
    # Ubah menjadi DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Standardisasi data menggunakan scaler bawaan Vian
    df_input_scaled = scaler.transform(df_input)
    
    # Prediksi menggunakan model Logistic Regression
    prediction = model.predict(df_input_scaled)[0]
    
    # Tampilkan Hasil Visual di Web
    st.subheader("Hasil Analisis Model:")
    if prediction == 0:
        st.error("❌ Hasil Prediksi: **UNSATISFIED** (Pelanggan Tidak Puas)")
    elif prediction == 1:
        st.warning("😐 Hasil Prediksi: **NEUTRAL** (Pelanggan Biasa Saja)")
    else:
        st.success("✅ Hasil Prediksi: **SATISFIED** (Pelanggan Sangat Puas!)")