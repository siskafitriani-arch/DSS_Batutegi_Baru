
import streamlit as st
import pandas as pd

st.title("DSS Bendungan Batutegi")

df = pd.read_excel("prediksi_hujan_15_hari.xlsx")

st.dataframe(df)
