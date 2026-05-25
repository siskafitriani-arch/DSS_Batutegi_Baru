import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from datetime import timedelta
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="DSS Bendungan Batutegi",
    layout="wide"
)

# ======================
# FILE
# ======================
MODEL_FILE = "model_lstm_batutegi (1).keras"
SCALER_X_FILE = "scaler_X (1).pkl"
SCALER_Y_FILE = "scaler_y (1).pkl"
DATASET_FILE = "dataset_lstm_batutegi_ready.xlsx"

VALIDASI_FILE = "hasil_prediksi_lstm (1).xlsx"
METRICS_FILE = "metrics_lstm.xlsx"

# ======================
# LOAD FILE
# ======================
@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_FILE)
    scaler_X = joblib.load(SCALER_X_FILE)
    scaler_y = joblib.load(SCALER_Y_FILE)
    return model, scaler_X, scaler_y

@st.cache_data
def load_dataset():
    hist = pd.read_excel(DATASET_FILE)
    hist.columns = [str(c).strip().lower() for c in hist.columns]
    hist["tanggal"] = pd.to_datetime(hist["tanggal"], errors="coerce")
    hist = hist.sort_values("tanggal").reset_index(drop=True)

    validasi = pd.read_excel(VALIDASI_FILE)
    validasi.columns = [str(c).strip().lower() for c in validasi.columns]

    try:
        metrics = pd.read_excel(METRICS_FILE)
        metrics.columns = [str(c).strip().lower() for c in metrics.columns]
    except:
        metrics = pd.DataFrame()

    return hist, validasi, metrics

try:
    model, scaler_X, scaler_y = load_artifacts()
    hist_df, validasi_df, metrics_df = load_dataset()
except Exception as e:
    st.error(f"Gagal memuat file: {e}")
    st.stop()

# ======================
# FITUR MODEL
# ======================
feature_cols = [
    "pos_hujan_1", "pos_hujan_2",
    "pos_hujan_1_lag_1", "pos_hujan_1_lag_2", "pos_hujan_1_lag_3",
    "pos_hujan_1_lag_4", "pos_hujan_1_lag_5", "pos_hujan_1_lag_6", "pos_hujan_1_lag_7",
    "pos_hujan_2_lag_1", "pos_hujan_2_lag_2", "pos_hujan_2_lag_3",
    "pos_hujan_2_lag_4", "pos_hujan_2_lag_5", "pos_hujan_2_lag_6", "pos_hujan_2_lag_7",
    "bulan", "time_index"
]

def clean_numeric(df, cols):
    for col in cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace("o", "0", regex=False)
            .str.replace("O", "0", regex=False)
            .str.replace("-", "0", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

hist_df = clean_numeric(hist_df, feature_cols)
hist_df = hist_df.dropna(subset=feature_cols + ["tanggal"]).reset_index(drop=True)

# ======================
# FUNGSI DSS
# ======================
def kategori_risiko(nilai):
    if nilai >= 50:
        return "Tinggi"
    elif nilai >= 20:
        return "Sedang"
    else:
        return "Rendah"

def rekomendasi_dss(kategori):
    if kategori == "Tinggi":
        return "SIAGA - Monitoring intensif inflow dan kesiapsiagaan operasional."
    elif kategori == "Sedang":
        return "WASPADA - Pantau kondisi DAS dan potensi kenaikan inflow."
    else:
        return "NORMAL - Operasi normal dan pemantauan rutin."

def prediksi_15_hari_lstm(tanggal_awal, rainfall_1_input, rainfall_2_input):
    tanggal_awal = pd.to_datetime(tanggal_awal)

    data_sebelum = hist_df[hist_df["tanggal"] <= tanggal_awal].copy()

    if len(data_sebelum) >= 7:
        base_df = data_sebelum.tail(7).copy()
    else:
        base_df = hist_df.tail(7).copy()

    last_row = base_df.iloc[-1].copy()

    r1_series = list(base_df["rainfall_1"].values) + [rainfall_1_input]
    r2_series = list(base_df["rainfall_2"].values) + [rainfall_2_input]

    current_time_index = int(last_row["time_index"]) if not pd.isna(last_row["time_index"]) else len(hist_df)

    hasil = []

    for i in range(1, 16):
        tanggal_pred = tanggal_awal + timedelta(days=i)

        row_input = {
            "rainfall_1": r1_series[-1],
            "rainfall_2": r2_series[-1],
            "bulan": tanggal_pred.month,
            "time_index": current_time_index
        }

        for lag in range(1, 8):
            row_input[f"rainfall_1_lag_{lag}"] = r1_series[-lag]
            row_input[f"rainfall_2_lag_{lag}"] = r2_series[-lag]

        X_input = pd.DataFrame([row_input])[feature_cols].values
        X_scaled = scaler_X.transform(X_input)
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

        pred_scaled = model.predict(X_scaled, verbose=0)
        pred = scaler_y.inverse_transform(pred_scaled)[0][0]
        pred = max(float(pred), 0)

        kategori = kategori_risiko(pred)

        hasil.append({
            "tanggal_prediksi": tanggal_pred,
            "hari_ke": f"H+{i}",
            "rainfall_1_input": round(rainfall_1_input, 2),
            "rainfall_2_input": round(rainfall_2_input, 2),
            "prediksi_hujan_lstm": round(pred, 2),
            "kategori_risiko": kategori,
            "rekomendasi": rekomendasi_dss(kategori)
        })

        r1_series.append(pred)
        r2_series.append(r2_series[-1])
        current_time_index += 1

    return pd.DataFrame(hasil)

# ======================
# STYLE
# ======================
st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: 800;
    color: white;
}
.subtitle {
    font-size: 17px;
    color: #b8c0cc;
    margin-bottom: 25px;
}
.card {
    background-color: #1e222d;
    padding: 22px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid #2e3440;
}
.card h3 {
    color: #9ca3af;
    font-size: 15px;
}
.card h2 {
    color: white;
    font-size: 28px;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)

# ======================
# SIDEBAR
# ======================
st.sidebar.title("DSS Batutegi")
st.sidebar.caption("LSTM + Hidrologi + Decision Support")

menu = st.sidebar.radio(
    "Menu",
    [
        "Overview",
        "Validasi Model",
        "Prediksi Interaktif",
        "Rekomendasi DSS",
        "Data"
    ]
)

# ======================
# VALIDASI KOLOM
# ======================
if "aktual" in validasi_df.columns:
    aktual_col = "aktual"
elif "aktual_dataset" in validasi_df.columns:
    aktual_col = "aktual_dataset"
else:
    aktual_col = None

if "prediksi" in validasi_df.columns:
    pred_col = "prediksi"
elif "prediksi_hujan_lstm" in validasi_df.columns:
    pred_col = "prediksi_hujan_lstm"
else:
    pred_col = None

# ======================
# INPUT INTERAKTIF
# ======================
st.sidebar.markdown("---")
st.sidebar.subheader("Input Prediksi Masa Depan")

tanggal_input = st.sidebar.date_input(
    "Tanggal awal prediksi",
    value=hist_df["tanggal"].max()
)

pos_hujan_1_input = st.sidebar.number_input(
    "Curah Hujan Pos 1 / pos_hujan_1 (mm)",
    min_value=0.0,
    max_value=300.0,
    value=float(hist_df["pos_hujan_1"].iloc[-1]),
    step=0.1
)

pos_hujan_2_input = st.sidebar.number_input(
    "Curah Hujan Pos 2 / pos_hujan_2 (mm)",
    min_value=0.0,
    max_value=300.0,
    value=float(hist_df["pos_hujan_2"].iloc[-1]),
    step=0.1
)

pred15_df = prediksi_15_hari_lstm(
    tanggal_input,
    pos_hujan_1_input,
    pos_hujan_2_input
)

max_pred = pred15_df["prediksi_hujan_lstm"].max()
mean_pred = pred15_df["prediksi_hujan_lstm"].mean()

if max_pred >= 50:
    status = "SIAGA"
elif max_pred >= 20:
    status = "WASPADA"
else:
    status = "NORMAL"

# ======================
# OVERVIEW
# ======================
if menu == "Overview":
    st.markdown('<div class="big-title">DSS Bendungan Batutegi</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Dashboard validasi model LSTM dan prediksi curah hujan 15 hari untuk dukungan operasi bendungan</div>',
        unsafe_allow_html=True
    )

    if aktual_col and pred_col:
        mae = abs(validasi_df[aktual_col] - validasi_df[pred_col]).mean()
        rmse = ((validasi_df[aktual_col] - validasi_df[pred_col]) ** 2).mean() ** 0.5
    else:
        mae = np.nan
        rmse = np.nan

    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f"""
    <div class="card">
        <h3>MAE Model</h3>
        <h2>{mae:.2f} mm</h2>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="card">
        <h3>RMSE Model</h3>
        <h2>{rmse:.2f} mm</h2>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="card">
        <h3>Prediksi Maksimum 15 Hari</h3>
        <h2>{max_pred:.2f} mm</h2>
    </div>
    """, unsafe_allow_html=True)

    col4.markdown(f"""
    <div class="card">
        <h3>Status DSS</h3>
        <h2>{status}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Grafik Prediksi Interaktif 15 Hari")

    fig = px.line(
        pred15_df,
        x="hari_ke",
        y="prediksi_hujan_lstm",
        markers=True,
        text="prediksi_hujan_lstm",
        title="Prediksi Curah Hujan 15 Hari ke Depan"
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        xaxis_title="Hari Prediksi",
        yaxis_title="Curah Hujan (mm)",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================
# VALIDASI MODEL
# ======================
elif menu == "Validasi Model":
    st.title("Validasi Model LSTM")

    st.subheader("Data Aktual dari Dataset")
    st.dataframe(
        hist_df[["tanggal", "rainfall_1", "rainfall_2"]],
        use_container_width=True
    )

    st.subheader("Grafik Aktual Dataset")
    fig = px.line(
        hist_df,
        x="tanggal",
        y=["rainfall_1", "rainfall_2"],
        title="Curah Hujan Aktual Dataset"
    )

    fig.update_layout(
        xaxis_title="Tanggal",
        yaxis_title="Curah Hujan (mm)",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)
# ======================
# PREDIKSI INTERAKTIF
# ======================
elif menu == "Prediksi Interaktif":
    st.title("Prediksi Curah Hujan 15 Hari ke Depan")

    st.write(f"Tanggal awal prediksi: **{pd.to_datetime(tanggal_input).date()}**")
    st.write(f"Input rainfall_1: **{rainfall_1_input:.2f} mm**")
    st.write(f"Input rainfall_2: **{rainfall_2_input:.2f} mm**")

    fig = px.bar(
        pred15_df,
        x="hari_ke",
        y="prediksi_hujan_lstm",
        color="kategori_risiko",
        text="prediksi_hujan_lstm",
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

# ======================
# REKOMENDASI DSS
# ======================
elif menu == "Rekomendasi DSS":
    st.title("Rekomendasi Operasi dan Pemeliharaan Bendungan")

    for _, row in pred15_df.iterrows():
        if row["kategori_risiko"] == "Tinggi":
            st.error(f"{row['hari_ke']} | {row['prediksi_hujan_lstm']} mm | {row['rekomendasi']}")
        elif row["kategori_risiko"] == "Sedang":
            st.warning(f"{row['hari_ke']} | {row['prediksi_hujan_lstm']} mm | {row['rekomendasi']}")
        else:
            st.success(f"{row['hari_ke']} | {row['prediksi_hujan_lstm']} mm | {row['rekomendasi']}")

# ======================
# DATA
# ======================
elif menu == "Data":
    st.title("Data Aktual Dataset")

    st.subheader("Data Aktual Curah Hujan dari Dataset")
    st.dataframe(
        hist_df[[
            "tanggal",
            "rainfall_1",
            "rainfall_2",
            "rainfall_1_lag_1",
            "rainfall_1_lag_2",
            "rainfall_1_lag_3",
            "rainfall_1_lag_7",
            "bulan",
            "time_index"
        ]],
        use_container_width=True
    )

    st.subheader("Grafik Aktual Curah Hujan Dataset")

    fig = px.line(
        hist_df,
        x="tanggal",
        y=["rainfall_1", "rainfall_2"],
        title="Data Aktual Curah Hujan dari Dataset"
    )

    fig.update_layout(
        xaxis_title="Tanggal",
        yaxis_title="Curah Hujan (mm)",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Data Prediksi Interaktif")
    st.dataframe(pred15_df, use_container_width=True)
