import streamlit as st
from sklearn.model_selection import train_test_split
from src.model_trainer import train_with_kfold, MODEL_MAP

def render(X, y):
    model_name = st.selectbox("Model", list(MODEL_MAP.keys()))

    if st.button("Train"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model, results = train_with_kfold(X_train, y_train, model_name)

        return model, results, X_test, y_test

    return None, None, None, None