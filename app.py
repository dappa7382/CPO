import streamlit as st
import pandas as pd
import numpy as np
from pytorch_forecasting import TemporalFusionTransformer
import lightning.pytorch as pl
from datetime import datetime, timedelta

# ==========================================
# 1. KONFIGURASI & LOAD MODEL
# ==========================================
st.set_page_config(page_title="CPO Prediction Pro", layout="wide")
st.title("🌴 CPO Price Forecaster (Manual Data Upload)")
st.markdown("""
Dashboard ini memprediksi harga CPO 7 hari ke depan. 
Silakan upload data historis terbaru dari Investing.com untuk memulai.
""")

@st.cache_resource
def load_model():
    return TemporalFusionTransformer.load_from_checkpoint("tft_cpo_model_final.ckpt")

model = load_model()

# ==========================================
# 2. FUNGSI PEMROSESAN DATA
# ==========================================
def process_uploaded_data(uploaded_df):
    # A. Load Data Cuaca dari Repo GitHub
    df_weather = pd.read_csv("historical_weather_data.csv")
    df_weather['Date'] = pd.to_datetime(df_weather['Date']).dt.normalize()
    
    # Hitung 30D Sum buat cuaca
    cuaca_cols = [col for col in df_weather.columns if 'Curah Hujan' in col]
    for col in cuaca_cols:
        nama_kolom_baru = f"{col}_30D_Sum" 
        df_weather[nama_kolom_baru] = df_weather[col].rolling(window=30).sum()
    
    # B. Rapihkan Data yang di-upload (Market Data)
    uploaded_df['Date'] = pd.to_datetime(uploaded_df['Date']).dt.normalize()
    uploaded_df = uploaded_df.sort_values('Date')
    
    # C. Gabungkan Market Data & Cuaca
    df_master = pd.merge(uploaded_df, df_weather, on='Date', how='left')
    
    # Tambal data kosong jika ada
    df_master = df_master.ffill().bfill()
    
    # D. Ambil 30 hari terakhir (Encoder)
    encoder_data = df_master.tail(30).copy()
    
    # E. Bikin baris masa depan 7 hari (Decoder)
    decoder_data = encoder_data.tail(7).copy()
    last_date = encoder_data['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
    decoder_data['Date'] = future_dates
    
    df_final = pd.concat([encoder_data, decoder_data], ignore_index=True)
    
    # F. Fitur Waktu & Group ID (WAJIB)
    df_final['Bulan'] = df_final['Date'].dt.month
    df_final['Kuartal'] = df_final['Date'].dt.quarter
    df_final['Hari_Seminggu'] = df_final['Date'].dt.dayofweek
    df_final['time_idx'] = list(range(1, len(df_final) + 1))
    df_final['group'] = "CPO" # Sesuai hasil training kamu
    
    return df_final

# ==========================================
# 3. INTERFACE & LOGIKA UTAMA
# ==========================================
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("📁 Upload Data")
    file_upload = st.file_uploader("Pilih file CSV hasil Investing.com", type=['csv'])
    
    if file_upload is not None:
        raw_data = pd.read_csv(file_upload)
        st.success("Data berhasil diunggah!")
        
        if st.button("🚀 Jalankan Prediksi", type="primary"):
            with st.spinner("AI sedang memproses data..."):
                try:
                    # 1. Olah data
                    df_input = process_uploaded_data(raw_data)
                    
                    # 2. Prediksi
                    future_preds = model.predict(df_input)
                    tebakan = np.array(future_preds).flatten()
                    
                    # 3. Simpan hasil
                    st.session_state['pred_ready'] = True
                    st.session_state['tebakan'] = tebakan
                    st.session_state['tanggal_depan'] = df_input.tail(7)['Date'].dt.date.values
                except Exception as e:
                    st.error(f"Terjadi kesalahan format: {e}")

with col2:
    st.subheader("📈 Hasil Analisa Harga")
    if st.session_state.get('pred_ready'):
        tebakan = st.session_state['tebakan']
        tanggal_depan = st.session_state['tanggal_depan']
        
        df_hasil = pd.DataFrame({
            "Tanggal": tanggal_depan,
            "Prediksi (MYR)": tebakan
        })
        
        # Metrik
        c1, c2 = st.columns(2)
        c1.metric("Prediksi Besok", f"{tebakan[0]:.0f} MYR")
        c2.metric("Puncak 7 Hari", f"{max(tebakan):.0f} MYR")
        
        # Grafik
        st.line_chart(df_hasil.set_index("Tanggal"))
        st.dataframe(df_hasil, use_container_width=True)
    else:
        st.info("Menunggu data diunggah untuk memulai prediksi.")
