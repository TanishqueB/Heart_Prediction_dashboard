import streamlit as st
from src.preprocessing import handle_zeros, encode_categoricals

def render(df):
    cols = st.multiselect("Columns to clean", df.columns)
    action = st.selectbox("Action", ["keep", "delete", "impute"])

    df = handle_zeros(df, cols, action)
    df = encode_categoricals(df)

    return df