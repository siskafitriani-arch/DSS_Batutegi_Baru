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

MODEL_FILE = "model_lstm_batutegi (1).keras"
SCALER_X_FILE = "scaler_X (1).pkl"
SCALER_Y_FILE = "scaler_y (1).pkl"
DATASET_FILE = "dataset_lstm_batutegi_ready.xlsx"

st.markdown("""
<style>
.big-title {
    font-size: 44px;
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
    font-size: 30px;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_FILE)
    scaler_X = joblib.load(SCALER_X_FILE)
    scaler_y = joblib.load(SCALER_Y_FILE)
    return model, scaler_X, scaler_y

@st.cache_data
def load_dataset():
    df = pd.read_excel(DATASET_FILE)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    df = df.sort_values("tanggal").reset_index(drop=True)
    return df

try:
    model, scaler_X, scaler_y = load_artifacts()
    hist_df = load_dataset()
except Exception as e:
    st.error(f"Gagal memuat model/scaler/dataset: {e}")
    st.stop()

feature_cols = [
    "rainfall_1", "rainfall_2",
    "rainfall_1_lag_1", "rainfall_1_lag_2", "rainfall_1_lag_3",
    "rainfall_1_lag_4", "rainfall_1_lag_5", "rainfall_1_lag_6", "rainfall_1_lag_7",
    "rainfall_2_lag_1", "rainfall_2_lag_2", "rainfall_2_lag_3",
    "rainfall_2_lag_4", "rainfall_2_lag_5", "rainfall_2_lag_6", "rainfall_2_lag_7",
    "bulan", "time_index"
]

missing_cols = [c for c in feature_cols + ["tanggal"] if c not in hist_df.columns]
if missing_cols:
    st.error(f"Kolom berikut tidak ada di dataset historis: {missing_cols}")
    st.stop()

def kategori_risiko(nilai):
    if nilai >= 50:
        return "Tinggi"
    elif nilai >= 20:
        return "Sedang"
    else:
        return "Rendah"

def rekomendasi_dss(kategori):
    if kategori == "Tinggi":
        return "Aktifkan peringatan dini. Monitoring intensif inflow dan koordinasi operasional bendungan."
    elif kategori == "Sedang":
        return "Pantau kondisi DAS, drainase, dan potensi kenaikan inflow. Siapkan mitigasi standar."
    else:
        return "Kondisi aman. Lanjutkan pemantauan rutin dan operasi normal."

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

def prediksi_15_hari_lstm(tanggal_awal, rainfall_1_input, rainfall_2_input):
    tanggal_awal = pd.to_datetime(tanggal_awal)

    data_sebelum = hist_df[hist_df["tanggal"] <= tanggal_awal].copy()

    if len(data_sebelum) >= 7:
        base_df = data_sebelum.tail(7).copy()
    else:
        base_df = hist_df.tail(7).copy()

    last_row = base_df.iloc[-1].copy()

    r1_hist = list(base_df["rainfall_1"].values)
    r2_hist = list(base_df["rainfall_2"].values)

    r1_series = r1_hist + [rainfall_1_input]
    r2_series = r2_hist + [rainfall_2_input]

    hasil = []

    current_time_index = int(last_row["time_index"]) if not pd.isna(last_row["time_index"]) else len(hist_df)

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

        aktual_row = hist_df[hist_df["tanggal"] == tanggal_pred]

        if len(aktual_row) > 0:
            aktual_dataset = float(aktual_row["rainfall_1"].iloc[0])
        else:
            aktual_dataset = np.nan

        hasil.append({
            "tanggal_prediksi": tanggal_pred,
            "hari_ke": f"H+{i}",
            "rainfall_1_input": round(rainfall_1_input, 2),
            "rainfall_2_input": round(rainfall_2_input, 2),
            "aktual_dataset": round(aktual_dataset, 2) if not pd.isna(aktual_dataset) else np.nan,
            "prediksi_hujan_lstm": round(pred, 2),
            "error": round(aktual_dataset - pred, 2) if not pd.isna(aktual_dataset) else np.nan,
            "kategori_risiko": kategori,
            "rekomendasi": rekomendasi_dss(kategori)
        })

        r1_series.append(pred)
        r2_series.append(r2_series[-1])
        current_time_index += 1

    return pd.DataFrame(hasil)

st.sidebar.title("DSS Batutegi")
st.sidebar.caption("LSTM + Hidrologi + Decision Support")

menu = st.sidebar.radio(
    "Menu",
    [
        "Input Prediksi",
        "Prediksi 15 Hari",
        "Analisis Risiko",
        "Rekomendasi DSS",
        "Data Historis"
    ]
)

st.sidebar.markdown("---")

tanggal_input = st.sidebar.date_input(
    "Tanggal awal prediksi",
    value=hist_df["tanggal"].max()
)

rainfall_1_input = st.sidebar.number_input(
    "Curah Hujan Pos 1 / rainfall_1 (mm)",
    min_value=0.0,
    max_value=300.0,
    value=float(hist_df["rainfall_1"].iloc[-1]),
    step=0.1
)

rainfall_2_input = st.sidebar.number_input(
    "Curah Hujan Pos 2 / rainfall_2 (mm)",
    min_value=0.0,
    max_value=300.0,
    value=float(hist_df["rainfall_2"].iloc[-1]),
    step=0.1
)

df_pred = prediksi_15_hari_lstm(
    tanggal_input,
    rainfall_1_input,
    rainfall_2_input
)

max_rain = df_pred["prediksi_hujan_lstm"].max()
avg_rain = df_pred["prediksi_hujan_lstm"].mean()
min_rain = df_pred["prediksi_hujan_lstm"].min()
mean_input = (rainfall_1_input + rainfall_2_input) / 2

if max_rain >= 50:
    status = "SIAGA"
elif max_rain >= 20:
    status = "WASPADA"
else:
    status = "NORMAL"

if menu == "Input Prediksi":
    st.markdown('<div class="big-title">DSS Bendungan Batutegi</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Prediksi curah hujan 15 hari ke depan menggunakan model LSTM asli dari dataset penelitian</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f"""
    <div class="card">
        <h3>Pos Hujan 1</h3>
        <h2>{rainfall_1_input:.2f} mm</h2>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="card">
        <h3>Pos Hujan 2</h3>
        <h2>{rainfall_2_input:.2f} mm</h2>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="card">
        <h3>Rata-rata Input</h3>
        <h2>{mean_input:.2f} mm</h2>
    </div>
    """, unsafe_allow_html=True)

    col4.markdown(f"""
    <div class="card">
        <h3>Status DSS</h3>
        <h2>{status}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Informasi Prediksi")
    st.write(f"Tanggal awal prediksi: **{pd.to_datetime(tanggal_input).date()}**")
    st.write(f"Periode prediksi: **{df_pred['tanggal_prediksi'].min().date()} s.d. {df_pred['tanggal_prediksi'].max().date()}**")

    st.markdown("### Grafik Prediksi Curah Hujan 15 Hari")

    fig = px.line(
        df_pred,
        x="hari_ke",
        y="prediksi_hujan_lstm",
        markers=True,
        text="prediksi_hujan_lstm",
        title="Prediksi Curah Hujan 15 Hari ke Depan Berbasis LSTM"
    )

    fig.update_traces(textposition="top center")

    fig.update_layout(
        xaxis_title="Hari Prediksi",
        yaxis_title="Curah Hujan (mm)",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Tabel Perbandingan Aktual Dataset dan Prediksi")
    st.dataframe(df_pred, use_container_width=True)

elif menu == "Prediksi 15 Hari":
    st.title("Prediksi Curah Hujan 15 Hari")

    fig = px.bar(
        df_pred,
        x="hari_ke",
        y="prediksi_hujan_lstm",
        color="kategori_risiko",
        text="prediksi_hujan_lstm",
        title="Prediksi Curah Hujan 15 Hari ke Depan"
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

    st.markdown("### Tabel Hasil Prediksi")
    st.dataframe(df_pred, use_container_width=True)

elif menu == "Analisis Risiko":
    st.title("Analisis Risiko Hidrologi")

    risiko_count = df_pred["kategori_risiko"].value_counts().reset_index()
    risiko_count.columns = ["kategori_risiko", "jumlah"]

    fig = px.pie(
        risiko_count,
        names="kategori_risiko",
        values="jumlah",
        title="Distribusi Kategori Risiko 15 Hari"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Prediksi Tertinggi")
    st.dataframe(
        df_pred.sort_values("prediksi_hujan_lstm", ascending=False).head(5),
        use_container_width=True
    )

elif menu == "Rekomendasi DSS":
    st.title("Rekomendasi Operasi dan Pemeliharaan Bendungan")

    for _, row in df_pred.iterrows():
        if row["kategori_risiko"] == "Tinggi":
            st.error(
                f"{row['tanggal_prediksi'].date()} | {row['hari_ke']} | "
                f"{row['prediksi_hujan_lstm']} mm - {row['rekomendasi']}"
            )
        elif row["kategori_risiko"] == "Sedang":
            st.warning(
                f"{row['tanggal_prediksi'].date()} | {row['hari_ke']} | "
                f"{row['prediksi_hujan_lstm']} mm - {row['rekomendasi']}"
            )
        else:
            st.success(
                f"{row['tanggal_prediksi'].date()} | {row['hari_ke']} | "
                f"{row['prediksi_hujan_lstm']} mm - {row['rekomendasi']}"
            )

elif menu == "Data Historis":
    st.title("Data Historis yang Digunakan Model")
    st.dataframe(hist_df.tail(100), use_container_width=True)
