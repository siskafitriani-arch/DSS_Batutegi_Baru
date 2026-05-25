import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="DSS Bendungan Batutegi",
    layout="wide"
)

# =========================
# FILE HASIL NOTEBOOK
# =========================
VALIDASI_FILE = "hasil_prediksi_lstm.xlsx"
PREDIKSI_15_FILE = "prediksi_hujan_15_hari.xlsx"
METRICS_FILE = "metrics_lstm.xlsx"

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    validasi = pd.read_excel(VALIDASI_FILE)
    pred15 = pd.read_excel(PREDIKSI_15_FILE)
    metrics = pd.read_excel(METRICS_FILE)

    validasi.columns = [str(c).strip().lower() for c in validasi.columns]
    pred15.columns = [str(c).strip().lower() for c in pred15.columns]
    metrics.columns = [str(c).strip().lower() for c in metrics.columns]

    return validasi, pred15, metrics

try:
    validasi_df, pred15_df, metrics_df = load_data()
except Exception as e:
    st.error(f"Gagal membaca file: {e}")
    st.stop()

# =========================
# DETEKSI KOLOM
# =========================
aktual_col = "aktual"
prediksi_col = "prediksi"

if aktual_col not in validasi_df.columns or prediksi_col not in validasi_df.columns:
    st.error("Kolom aktual/prediksi tidak ditemukan di hasil_prediksi_lstm.xlsx")
    st.write("Kolom tersedia:", validasi_df.columns.tolist())
    st.stop()

# kolom prediksi 15 hari
if "prediksi_hujan_lstm" in pred15_df.columns:
    pred15_col = "prediksi_hujan_lstm"
elif "prediksi_curah_hujan" in pred15_df.columns:
    pred15_col = "prediksi_curah_hujan"
else:
    st.error("Kolom prediksi 15 hari tidak ditemukan.")
    st.write("Kolom tersedia:", pred15_df.columns.tolist())
    st.stop()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("DSS Batutegi")
menu = st.sidebar.radio(
    "Menu",
    [
        "Overview",
        "Validasi Model",
        "Prediksi 15 Hari",
        "Rekomendasi DSS",
        "Data"
    ]
)

# =========================
# OVERVIEW
# =========================
if menu == "Overview":
    st.title("DSS Bendungan Batutegi")
    st.caption("Dashboard validasi model LSTM dan prediksi curah hujan 15 hari")

    mae = abs(validasi_df[aktual_col] - validasi_df[prediksi_col]).mean()
    rmse = ((validasi_df[aktual_col] - validasi_df[prediksi_col]) ** 2).mean() ** 0.5
    pred_max = pred15_df[pred15_col].max()

    if pred_max >= 50:
        status = "SIAGA"
    elif pred_max >= 20:
        status = "WASPADA"
    else:
        status = "NORMAL"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{mae:.2f} mm")
    c2.metric("RMSE", f"{rmse:.2f} mm")
    c3.metric("Prediksi Maksimum 15 Hari", f"{pred_max:.2f} mm")
    c4.metric("Status DSS", status)

    st.subheader("Grafik Aktual vs Prediksi")
    fig = px.line(
        validasi_df,
        y=[aktual_col, prediksi_col],
        title="Perbandingan Aktual Dataset dan Prediksi LSTM"
    )
    fig.update_layout(
        xaxis_title="Data Uji",
        yaxis_title="Curah Hujan (mm)",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# VALIDASI MODEL
# =========================
elif menu == "Validasi Model":
    st.title("Validasi Model LSTM")

    validasi_df["error"] = validasi_df[aktual_col] - validasi_df[prediksi_col]

    fig = px.line(
        validasi_df,
        y=[aktual_col, prediksi_col],
        title="Aktual Dataset vs Hasil Prediksi LSTM"
    )
    fig.update_layout(
        xaxis_title="Data Uji",
        yaxis_title="Curah Hujan (mm)",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tabel Validasi")
    st.dataframe(validasi_df, use_container_width=True)

# =========================
# PREDIKSI 15 HARI
# =========================
elif menu == "Prediksi 15 Hari":
    st.title("Prediksi Curah Hujan 15 Hari ke Depan")

    if "hari_ke" not in pred15_df.columns:
        pred15_df["hari_ke"] = [f"H+{i}" for i in range(1, len(pred15_df) + 1)]

    fig = px.bar(
        pred15_df,
        x="hari_ke",
        y=pred15_col,
        text=pred15_col,
        title="Prediksi Curah Hujan 15 Hari"
    )
    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )
    fig.update_layout(
        xaxis_title="Hari Prediksi",
        yaxis_title="Curah Hujan (mm)",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(pred15_df, use_container_width=True)

# =========================
# REKOMENDASI DSS
# =========================
elif menu == "Rekomendasi DSS":
    st.title("Rekomendasi DSS Operasi Bendungan")

    if "hari_ke" not in pred15_df.columns:
        pred15_df["hari_ke"] = [f"H+{i}" for i in range(1, len(pred15_df) + 1)]

    for _, row in pred15_df.iterrows():
        hujan = row[pred15_col]

        if hujan >= 50:
            st.error(f"{row['hari_ke']} | {hujan:.2f} mm | SIAGA - Monitoring intensif inflow dan kesiapsiagaan operasional.")
        elif hujan >= 20:
            st.warning(f"{row['hari_ke']} | {hujan:.2f} mm | WASPADA - Pantau kondisi DAS dan potensi kenaikan inflow.")
        else:
            st.success(f"{row['hari_ke']} | {hujan:.2f} mm | NORMAL - Operasi normal dan pemantauan rutin.")

# =========================
# DATA
# =========================
elif menu == "Data":
    st.title("Data Dashboard")

    st.subheader("Data Validasi Model")
    st.dataframe(validasi_df, use_container_width=True)

    st.subheader("Data Prediksi 15 Hari")
    st.dataframe(pred15_df, use_container_width=True)

    st.subheader("Metrik Model")
    st.dataframe(metrics_df, use_container_width=True)
