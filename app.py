
import streamlit as st
import pandas as pd

st.title("DSS Bendungan Batutegi")

df = pd.read_excel("prediksi_hujan.csv")

st.dataframe(df)
