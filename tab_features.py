import streamlit as st
from src.feature_selection import apply_selection

def render(df, target):
    method = st.selectbox("Method", ["all", "variance", "infogain"])

    features, X, y = apply_selection(df, target, method)

    st.write("Features:", features)

    return X, y