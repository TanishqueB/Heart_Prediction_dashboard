import streamlit as st
from src.data_loader import load_data
from app.tabs import tab_eda, tab_cleaning, tab_features, tab_training, tab_performance

st.set_page_config(layout="wide")

df = load_data("data/heart_disease.csv")

tabs = st.tabs(["EDA", "Cleaning", "Features", "Training", "Performance"])

with tabs[0]:
    target = tab_eda.render(df)

with tabs[1]:
    df_clean = tab_cleaning.render(df)

with tabs[2]:
    X, y = tab_features.render(df_clean, target)

with tabs[3]:
    model, results, X_test, y_test = tab_training.render(X, y)

with tabs[4]:
    tab_performance.render(model, results, X_test, y_test)