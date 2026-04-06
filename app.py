import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pytorch_forecasting import TemporalFusionTransformer
import lightning.pytorch as pl
from datetime import datetime, timedelta

# ==========================================
# 1. KONFIGURASI & LOAD MODEL
# ==========================================
st.set_page_config(page_title="CPO Intelligence Dashboard", layout="wide")
st.title("🌴 CPO Price Forecasting Dashboard (Hybrid Mode)")

@st.cache_resource
def load_model():
    # Pastikan file .ckpt hasil training kemarin ada di folder yang sama
    return TemporalFusionTransformer.load_from_checkpoint("tft_cpo_model_final.ckpt")

model = load_model()

# ==========================================
# 2. FUNGSI PENYIAPAN DATA (THE HYBRID ENGINE)
# ==========================================
@st.cache_data(ttl=3600) # Cache 1 jam biar kita gak diblokir Yahoo
def fetch_and_prepare_data():
    # A. Tarik Data Bursa (Live)
    tickers = {
        "Price Palm Oil": "FCPO.KL",
        "Price Soybean Oil": "ZL=F",
        "Price Crude Oil": "CL=F",
        "USD/MYR": "MYR=X",
        "USD/IDR": "IDR=X"
    }
    
    df_list = []
    for name, ticker in tickers.items():
        # CARA BARU YANG LEBIH TAHAN BANTING
        ticker_obj = yf.Ticker(ticker)
        df_temp = ticker_obj.history(period="60d")['Close'].reset_index()
        
        # Jaga-jaga kalau format yfinance berubah
        if 'Date' not in df_temp.columns and 'Datetime' in df_temp.columns:
            df_temp.rename(columns={'Datetime': 'Date'}, inplace=True)
            
        df_temp = df_temp[['Date', 'Close']].rename(columns={'Close': name})
        
        # Bersihkan zona waktu & bulatkan ke tengah malam (00:00:00) biar sinkron
        df_temp['Date'] = pd.to_datetime(df_temp['Date']).dt.tz_localize(None).dt.normalize()
        
        df_list.append(df_temp)
        
    # Gabungin semua data harga
    df_finance = df_list[0]
    for i in range(1, len(df_list)):
        df_finance = pd.merge(df_finance, df_list[i], on='Date', how='outer')
        
    df_finance.sort_values('Date', inplace=True)
    
    # --- FIX ERROR NaT DI SINI ---
    # Tambal data kosong ke bawah (ffill), lalu tambal ke atas (bfill)
    df_finance.ffill(inplace=True)
    df_finance.bfill(inplace=True)
    # df_finance.dropna() KITA BUANG! Biar dataframe lu gak pernah kosong lagi
    
    # B. Load Data Cuaca (Historical/Static)
    df_weather = pd.read_csv("historical_weather_data.csv")
    df_weather['Date'] = pd.to_datetime(df_weather['Date']).dt.normalize()
    
    # Hitung 30D Sum buat cuaca
    cuaca_cols = [col for col in df_weather.columns if 'Curah Hujan' in col]
    for col in cuaca_cols:
        nama_kolom_baru = f"{col}_30D_sum" 
        df_weather[nama_kolom_baru] = df_weather[col].rolling(window=30).sum()
        
    # C. Gabungkan Harga & Cuaca
    df_master = pd.merge(df_finance, df_weather, on='Date', how='left')
    df_master.ffill(inplace=True)
    df_master.bfill(inplace=True) # Pengaman ganda
    
    # Ambil 30 baris terakhir buat ancang-ancang
    encoder_data = df_master.tail(30).copy()
    
    # D. Bikin baris masa depan 7 hari
    decoder_data = encoder_data.tail(7).copy()
    last_date = encoder_data['Date'].max()
    
    # --- PENGAMAN TERAKHIR ---
    # Kalau sampai last_date masih error (misal internet down), set ke hari ini
    if pd.isna(last_date):
        last_date = pd.Timestamp.today().normalize()
        
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
    decoder_data['Date'] = future_dates
    
    df_final = pd.concat([encoder_data, decoder_data], ignore_index=True)
    
    # E. Bersihkan Fitur Waktu
    df_final['Bulan'] = df_final['Date'].dt.month
    df_final['Kuartal'] = df_final['Date'].dt.quarter
    df_final['Hari_Seminggu'] = df_final['Date'].dt.dayofweek
    df_final['time_idx'] = list(range(1, len(df_final) + 1))
    df_final['group'] = 0 
    
    return df_final
# ==========================================
# 3. INTERFACE DASHBOARD
# ==========================================
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Control Panel")
    if st.button("🚀 Run Live Prediction", type="primary"):
        with st.spinner("Menarik data bursa dan menghitung prediksi..."):
            
            # 1. Siapkan Data Input
            df_input = fetch_and_prepare_data()
            
            # 2. Jalankan Prediksi Pakai Model AI
            future_preds = model.predict(df_input, mode="raw", return_x=True)
            
            # 3. Ekstrak Hasil
            tebakan = future_preds.output.numpy()[0, :, 3]
            tanggal_depan = pd.date_range(start=pd.Timestamp.today(), periods=7).date
            
            # 4. Simpan ke Session State biar gak hilang pas refresh
            st.session_state['pred_ready'] = True
            st.session_state['tebakan'] = tebakan
            st.session_state['tanggal_depan'] = tanggal_depan
            
            st.success("Prediksi Selesai!")

with col2:
    st.subheader("Price Prediction Analysis")
    if st.session_state.get('pred_ready'):
        
        # Panggil data dari Session State
        tebakan = st.session_state['tebakan']
        tanggal_depan = st.session_state['tanggal_depan']
        
        df_hasil = pd.DataFrame({
            "Tanggal": tanggal_depan,
            "Prediksi Harga (MYR)": tebakan
        })
        
        # Hitung selisih untuk tren
        selisih = tebakan[-1] - tebakan[0]
        tren_teks = "Naik" if selisih > 0 else "Turun"
        
        col2_a, col2_b = st.columns(2)
        with col2_a:
            st.metric("Prediksi Besok", f"{tebakan[0]:.0f} MYR", delta=f"{selisih:.0f} MYR (Tren 7 Hari)")
        with col2_b:
            st.metric("Puncak Tertinggi 7 Hari", f"{max(tebakan):.0f} MYR")
            
        st.markdown("### Tren Harga 7 Hari Kedepan")
        st.line_chart(df_hasil.set_index("Tanggal"))
        
        st.markdown("### Rincian Angka Harian")
        st.dataframe(df_hasil, use_container_width=True)

    else:
        st.info("Klik tombol 'Run Live Prediction' di sebelah kiri untuk memulai analisa.")

# ==========================================
# 4. EXPLAINABLE AI (XAI) SECTION (COMING SOON)
# ==========================================
st.divider()
st.subheader("Why is the price moving?")
st.write("*Insight Extracting Engine is currently being optimized. Variables importance will be displayed here.*")
