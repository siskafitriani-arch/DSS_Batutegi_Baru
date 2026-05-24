import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="DSS Bendungan Batutegi",
    layout="wide"
)

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

def buat_prediksi_15_hari(hujan_hari_ini):
    hasil = []
    nilai_awal = hujan_hari_ini

    for i in range(1, 16):
        faktor_decay = 0.88 ** i
        variasi_musiman = 3 * np.sin(i / 2)

        prediksi = (nilai_awal * faktor_decay) + variasi_musiman

        if hujan_hari_ini >= 50 and i <= 3:
            prediksi += 8
        elif hujan_hari_ini >= 20 and i <= 3:
            prediksi += 4

        prediksi = max(prediksi, 0)

        kategori = kategori_risiko(prediksi)

        hasil.append({
            "hari_ke": f"H+{i}",
            "prediksi_curah_hujan": round(prediksi, 2),
            "kategori_risiko": kategori,
            "rekomendasi": rekomendasi_dss(kategori)
        })

    return pd.DataFrame(hasil)

st.sidebar.title("DSS Batutegi")
menu = st.sidebar.radio(
    "Menu",
    [
        "Input Manual",
        "Prediksi 15 Hari",
        "Analisis Risiko",
        "Rekomendasi DSS"
    ]
)

st.sidebar.markdown("---")
hujan_input = st.sidebar.number_input(
    "Input Curah Hujan Hari Ini (mm)",
    min_value=0.0,
    max_value=300.0,
    value=25.0,
    step=0.1
)

df = buat_prediksi_15_hari(hujan_input)

max_rain = df["prediksi_curah_hujan"].max()
avg_rain = df["prediksi_curah_hujan"].mean()
min_rain = df["prediksi_curah_hujan"].min()

if max_rain >= 50:
    status = "SIAGA"
elif max_rain >= 20:
    status = "WASPADA"
else:
    status = "NORMAL"

if menu == "Input Manual":
    st.markdown('<div class="big-title">DSS Bendungan Batutegi</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Prediksi curah hujan 15 hari ke depan berdasarkan input manual curah hujan hari ini</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f"""
    <div class="card">
        <h3>Input Hujan Hari Ini</h3>
        <h2>{hujan_input:.2f} mm</h2>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="card">
        <h3>Prediksi Maksimum</h3>
        <h2>{max_rain:.2f} mm</h2>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="card">
        <h3>Rata-rata Prediksi</h3>
        <h2>{avg_rain:.2f} mm</h2>
    </div>
    """, unsafe_allow_html=True)

    col4.markdown(f"""
    <div class="card">
        <h3>Status DSS</h3>
        <h2>{status}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Grafik Prediksi Curah Hujan 15 Hari")

    fig = px.line(
        df,
        x="hari_ke",
        y="prediksi_curah_hujan",
        markers=True,
        title="Prediksi Curah Hujan 15 Hari ke Depan"
    )
    fig.update_layout(
        xaxis_title="Hari Prediksi",
        yaxis_title="Curah Hujan (mm)",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Prediksi 15 Hari":
    st.title("Prediksi Curah Hujan 15 Hari")

    fig = px.bar(
        df,
        x="hari_ke",
        y="prediksi_curah_hujan",
        color="kategori_risiko",
        title="Prediksi Curah Hujan dan Kategori Risiko"
    )
    fig.update_layout(
        xaxis_title="Hari Prediksi",
        yaxis_title="Curah Hujan (mm)",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df, use_container_width=True)

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

    st.markdown("### Prediksi Tertinggi")
    st.dataframe(
        df.sort_values("prediksi_curah_hujan", ascending=False).head(5),
        use_container_width=True
    )

elif menu == "Rekomendasi DSS":
    st.title("Rekomendasi Operasi dan Pemeliharaan Bendungan")

    for _, row in df.iterrows():
        if row["kategori_risiko"] == "Tinggi":
            st.error(f"{row['hari_ke']}: {row['prediksi_curah_hujan']} mm - {row['rekomendasi']}")
        elif row["kategori_risiko"] == "Sedang":
            st.warning(f"{row['hari_ke']}: {row['prediksi_curah_hujan']} mm - {row['rekomendasi']}")
        else:
            st.success(f"{row['hari_ke']}: {row['prediksi_curah_hujan']} mm - {row['rekomendasi']}")
