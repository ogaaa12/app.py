import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Load Model
@st.cache_resource
def load_model():
    with open('model_diamond.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Judul Aplikasi
st.title("💎 Diamond Price Predictor")
st.markdown("Masukkan spesifikasi berlian di bawah ini untuk mengestimasi harga.")

# 2. Membuat Layout Input User
col1, col2 = st.columns(2)

with col1:
    carat = st.number_input("Carat (Berat)", min_value=0.1, max_value=5.0, value=0.7, step=0.01)
    depth = st.number_input("Depth %", min_value=40.0, max_value=80.0, value=61.0, step=0.1)
    table = st.number_input("Table %", min_value=40.0, max_value=90.0, value=57.0, step=0.1)

with col2:
    x = st.number_input("Length (x) in mm", min_value=0.1, max_value=15.0, value=5.5)
    y = st.number_input("Width (y) in mm", min_value=0.1, max_value=15.0, value=5.5)
    z = st.number_input("Depth (z) in mm", min_value=0.1, max_value=15.0, value=3.5)

st.divider()

# Input Kategorikal
col3, col4, col5 = st.columns(3)

with col3:
    cut = st.selectbox("Cut", ["Ideal", "Premium", "Good", "Very Good", "Fair"])
with col4:
    color = st.selectbox("Color", ["E", "I", "J", "H", "F", "G", "D"])
with col5:
    clarity = st.selectbox("Clarity", ["SI2", "SI1", "VS1", "VS2", "VVS2", "VVS1", "I1", "IF"])

# 3. Preprocessing Input (Harus sesuai dengan One-Hot Encoding saat training)
def preprocess_input():
    # Buat dataframe awal dengan nilai 0
    # Catatan: Kolom harus persis sama dengan X_train.columns (kecuali drop_first=True)
    data = {
        'carat': [carat], 'depth': [depth], 'table': [table], 'x': [x], 'y': [y], 'z': [z],
        # Cut (Sesuai drop_first=True, 'Fair' hilang)
        'cut_Good': [1 if cut == 'Good' else 0],
        'cut_Ideal': [1 if cut == 'Ideal' else 0],
        'cut_Premium': [1 if cut == 'Premium' else 0],
        'cut_Very Good': [1 if cut == 'Very Good' else 0],
        # Color (Sesuai drop_first=True, 'D' hilang)
        'color_E': [1 if color == 'E' else 0],
        'color_F': [1 if color == 'F' else 0],
        'color_G': [1 if color == 'G' else 0],
        'color_H': [1 if color == 'H' else 0],
        'color_I': [1 if color == 'I' else 0],
        'color_J': [1 if color == 'J' else 0],
        # Clarity (Sesuai drop_first=True, 'I1' hilang)
        'clarity_IF': [1 if clarity == 'IF' else 0],
        'clarity_SI1': [1 if clarity == 'SI1' else 0],
        'clarity_SI2': [1 if clarity == 'SI2' else 0],
        'clarity_VS1': [1 if clarity == 'VS1' else 0],
        'clarity_VS2': [1 if clarity == 'VS2' else 0],
        'clarity_VVS1': [1 if clarity == 'VVS1' else 0],
        'clarity_VVS2': [1 if clarity == 'VVS2' else 0],
    }
    return pd.DataFrame(data)

input_df = preprocess_input()

# 4. Prediksi
if st.button("Prediksi Harga"):
    prediction = model.predict(input_df)
    st.success(f"Estimasi Harga Berlian: **${prediction[0]:,.2f}**")
    
    # Tambahan info
    st.info("Catatan: Prediksi ini menggunakan model XGBoost yang telah dilatih.")
