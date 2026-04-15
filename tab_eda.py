import streamlit as st
from src.data_loader import get_summary

def render(df):
    st.write("Columns:", df.columns)

    target = st.selectbox("Select Target", df.columns)

    st.dataframe(get_summary(df))

    return target