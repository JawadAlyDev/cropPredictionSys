import streamlit as st
import pandas as pd

st.title("User CSV upload")

uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
if uploaded_file is not None:
    st.success("File uploaded successfully!")
    df = pd.read_csv(uploaded_file)
    st.write("### First 5 rows of the Uploaded CSV file:")
    st.dataframe(df.head())

    