import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="DSS Bendungan Batutegi",
    layout="wide"
)

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("prediksi_curah_hujan.csv")

# =====================
# STYLE
# =====================
st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: 800;
    color: white;
}
.card {
    background-color: #1e222d;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}
.card h3 {
    color: #9ca3af;
    font-size: 16px;
}
.card h2 {
    color: white;
    font-size: 30px;
}
.status-normal {
    background-color: #14532d;
    padding: 18px;
    border-radius: 12px;
    color: white;
    font-size: 22px;
    font-weight: bold;
}
.status-waspada {
    background-color: #854d0e;
    padding: 18px;
    border-radius: 12px;
    color: white;
    font-size: 22px;
    font-weight: bold;
}
.status-siaga {
    background-color: #7f1d1d;
    padding: 18px;
    border-radius: 12px;
    color: white;
    font-size: 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =====================
# SIDEBAR
# =====================
st.sidebar.title("DSS Batutegi")
menu = st.sidebar.radio(
    "Menu",
    [
        "Overview",
        "Prediksi 15 Hari",
        "Analisis Risiko",
        "Rekomendasi DSS"
    ]
)

# pastikan urutan
df = df.reset_index(drop=True)

# ambil data terbaru / tertinggi
latest = df.iloc[-1]
max_rain = df["curah_hujan"].max()
avg_rain = df["curah_hujan"].mean()

# status utama
if max_rain >= 50:
    status = "SIAGA"
elif max_rain >= 20:
    status = "WASPADA"
else:
    status = "NORMAL"

# =====================
# OVERVIEW
# =====================
if menu == "Overview":
    st.markdown('<div class="big-title">DSS Bendungan Batutegi</div>', unsafe_allow_html=True)
    st.caption("Dashboard prediksi curah hujan harian berbasis LSTM untuk mendukung operasi dan pemeliharaan bendungan")

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
        <h3>Jumlah Hari Prediksi</h3>
        <h2>{len(df)} hari</h2>
    </div>
    """, unsafe_allow_html=True)

    col4.markdown(f"""
    <div class="card">
        <h3>Status DSS</h3>
        <h2>{status}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Status Operasional")

    if status == "SIAGA":
        st.markdown('<div class="status-siaga">SIAGA - Potensi hujan tinggi, perlu monitoring intensif.</div>', unsafe_allow_html=True)
    elif status == "WASPADA":
        st.markdown('<div class="status-waspada">WASPADA - Pantau kondisi DAS dan area rawan genangan.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-normal">NORMAL - Kondisi relatif aman.</div>', unsafe_allow_html=True)

    st.markdown("### Grafik Prediksi Curah Hujan 15 Hari")

    fig = px.line(
        df,
        x=df.index + 1,
        y="curah_hujan",
        markers=True,
        title="Prediksi Curah Hujan 15 Hari ke Depan"
    )
    fig.update_layout(
        xaxis_title="Hari ke-",
        yaxis_title="Curah Hujan (mm)",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================
# PREDIKSI 15 HARI
# =====================
elif menu == "Prediksi 15 Hari":
    st.title("Prediksi Curah Hujan 15 Hari")

    fig = px.bar(
        df,
        x=df.index + 1,
        y="curah_hujan",
        color="kategori_risiko",
        title="Prediksi Curah Hujan dan Kategori Risiko"
    )
    fig.update_layout(
        xaxis_title="Hari ke-",
        yaxis_title="Curah Hujan (mm)",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df, use_container_width=True)

# =====================
# ANALISIS RISIKO
# =====================
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

    st.markdown("### Hari dengan Risiko Tertinggi")
    st.dataframe(
        df.sort_values("curah_hujan", ascending=False).head(5),
        use_container_width=True
    )

# =====================
# REKOMENDASI DSS
# =====================
elif menu == "Rekomendasi DSS":
    st.title("Rekomendasi Operasi dan Pemeliharaan Bendungan")

    for _, row in df.iterrows():
        if row["kategori_risiko"] == "Tinggi":
            st.error(f"Hari ke-{row.name+1}: {row['curah_hujan']} mm - {row['rekomendasi']}")
        elif row["kategori_risiko"] == "Sedang":
            st.warning(f"Hari ke-{row.name+1}: {row['curah_hujan']} mm - {row['rekomendasi']}")
        else:
            st.success(f"Hari ke-{row.name+1}: {row['curah_hujan']} mm - {row['rekomendasi']}")
