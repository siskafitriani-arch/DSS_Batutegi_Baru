import streamlit as st
import pandas as pd
import plotly.express as px

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="DSS Bendungan Batutegi",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================
DATA_FILE = "prediksi_curah_hujan.csv"

try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    st.error(f"File {DATA_FILE} tidak ditemukan. Pastikan file sudah ada di GitHub.")
    st.stop()

df.columns = [str(c).strip().lower() for c in df.columns]

# =====================================================
# DETEKSI KOLOM HUJAN OTOMATIS
# =====================================================
possible_rain_cols = [
    "curah_hujan",
    "prediksi_hujan_lstm",
    "prediksi_curah_hujan",
    "prediksi_hujan",
    "rainfall",
    "rainfall_1"
]

rain_col = None
for col in possible_rain_cols:
    if col in df.columns:
        rain_col = col
        break

if rain_col is None:
    st.error("Kolom curah hujan tidak ditemukan.")
    st.write("Kolom yang tersedia:", df.columns.tolist())
    st.stop()

# =====================================================
# BUAT KATEGORI RISIKO JIKA BELUM ADA
# =====================================================
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

if "kategori_risiko" not in df.columns:
    df["kategori_risiko"] = df[rain_col].apply(kategori_risiko)

if "rekomendasi" not in df.columns:
    df["rekomendasi"] = df["kategori_risiko"].apply(rekomendasi_dss)

# =====================================================
# STYLE CSS
# =====================================================
st.markdown("""
<style>
.big-title {
    font-size: 44px;
    font-weight: 800;
    color: white;
    margin-bottom: 5px;
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
    margin-bottom: 10px;
}
.card h2 {
    color: white;
    font-size: 30px;
    font-weight: 800;
}
.status-normal {
    background-color: #14532d;
    padding: 20px;
    border-radius: 14px;
    color: white;
    font-size: 22px;
    font-weight: bold;
}
.status-waspada {
    background-color: #854d0e;
    padding: 20px;
    border-radius: 14px;
    color: white;
    font-size: 22px;
    font-weight: bold;
}
.status-siaga {
    background-color: #7f1d1d;
    padding: 20px;
    border-radius: 14px;
    color: white;
    font-size: 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("DSS Batutegi")
st.sidebar.caption("Dashboard Prediksi Curah Hujan Harian")

menu = st.sidebar.radio(
    "Menu",
    [
        "Overview",
        "Prediksi 15 Hari",
        "Analisis Risiko",
        "Rekomendasi DSS",
        "Data"
    ]
)

# =====================================================
# HITUNG RINGKASAN
# =====================================================
max_rain = df[rain_col].max()
avg_rain = df[rain_col].mean()
min_rain = df[rain_col].min()
jumlah_hari = len(df)

if max_rain >= 50:
    status = "SIAGA"
elif max_rain >= 20:
    status = "WASPADA"
else:
    status = "NORMAL"

# =====================================================
# HALAMAN OVERVIEW
# =====================================================
if menu == "Overview":
    st.markdown('<div class="big-title">DSS Bendungan Batutegi</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Dashboard prediksi curah hujan harian berbasis LSTM untuk mendukung operasi dan pemeliharaan bendungan</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f"""
    <div class="card">
        <h3>Prediksi Maksimum</h3>
        <h2>{max_rain:.2f} mm</h2>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="card">
        <h3>Rata-rata Prediksi</h3>
        <h2>{avg_rain:.2f} mm</h2>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="card">
        <h3>Prediksi Minimum</h3>
        <h2>{min_rain:.2f} mm</h2>
    </div>
    """, unsafe_allow_html=True)

    col4.markdown(f"""
    <div class="card">
        <h3>Jumlah Hari</h3>
        <h2>{jumlah_hari} hari</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Status DSS")

    if status == "SIAGA":
        st.markdown(
            '<div class="status-siaga">SIAGA - Potensi hujan tinggi. Perlu monitoring intensif dan kesiapsiagaan operasional.</div>',
            unsafe_allow_html=True
        )
    elif status == "WASPADA":
        st.markdown(
            '<div class="status-waspada">WASPADA - Terdapat potensi hujan sedang. Pantau kondisi DAS dan inflow.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-normal">NORMAL - Kondisi prediksi curah hujan relatif rendah.</div>',
            unsafe_allow_html=True
        )

    st.markdown("### Grafik Prediksi Curah Hujan")

    fig = px.line(
        df,
        x=df.index + 1,
        y=rain_col,
        markers=True,
        title="Prediksi Curah Hujan 15 Hari ke Depan"
    )
    fig.update_layout(
        xaxis_title="Hari ke-",
        yaxis_title="Curah Hujan (mm)",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# HALAMAN PREDIKSI 15 HARI
# =====================================================
elif menu == "Prediksi 15 Hari":
    st.title("Prediksi Curah Hujan 15 Hari")

    fig = px.bar(
        df,
        x=df.index + 1,
        y=rain_col,
        color="kategori_risiko",
        title="Prediksi Curah Hujan dan Kategori Risiko"
    )
    fig.update_layout(
        xaxis_title="Hari ke-",
        yaxis_title="Curah Hujan (mm)",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Tabel Prediksi")
    st.dataframe(df, use_container_width=True)

# =====================================================
# HALAMAN ANALISIS RISIKO
# =====================================================
elif menu == "Analisis Risiko":
    st.title("Analisis Risiko Hidrologi")

    risiko_count = df["kategori_risiko"].value_counts().reset_index()
    risiko_count.columns = ["kategori_risiko", "jumlah"]

    fig = px.pie(
        risiko_count,
        names="kategori_risiko",
        values="jumlah",
        title="Distribusi Kategori Risiko"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Hari dengan Curah Hujan Tertinggi")
    st.dataframe(
        df.sort_values(rain_col, ascending=False).head(5),
        use_container_width=True
    )

# =====================================================
# HALAMAN REKOMENDASI DSS
# =====================================================
elif menu == "Rekomendasi DSS":
    st.title("Rekomendasi Operasi dan Pemeliharaan Bendungan")

    for i, row in df.iterrows():
        nilai_hujan = row[rain_col]
        kategori = row["kategori_risiko"]
        rekomendasi = row["rekomendasi"]

        if kategori == "Tinggi":
            st.error(f"Hari ke-{i+1}: {nilai_hujan:.2f} mm - {rekomendasi}")
        elif kategori == "Sedang":
            st.warning(f"Hari ke-{i+1}: {nilai_hujan:.2f} mm - {rekomendasi}")
        else:
            st.success(f"Hari ke-{i+1}: {nilai_hujan:.2f} mm - {rekomendasi}")

# =====================================================
# HALAMAN DATA
# =====================================================
elif menu == "Data":
    st.title("Data Prediksi")
    st.write("Kolom yang terbaca:", df.columns.tolist())
    st.dataframe(df, use_container_width=True)
