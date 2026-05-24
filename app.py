
import streamlit as st
import pandas as pd

st.title("DSS Bendungan Batutegi")

df = pd.read_csv("prediksi_curah_hujan.csv")

st.dataframe(df)
