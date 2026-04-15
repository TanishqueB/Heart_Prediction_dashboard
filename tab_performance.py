import streamlit as st
from src.evaluator import compute_avg_metrics, plot_confusion, plot_roc, get_report

def render(model, results, X_test, y_test):
    if model is None:
        st.warning("Train model first")
        return

    metrics = compute_avg_metrics(results)
    st.write(metrics)

    y_pred = model.predict(X_test)

    st.pyplot(plot_confusion(y_test, y_pred))
    st.pyplot(plot_roc(model, X_test, y_test))
    st.dataframe(get_report(y_test, y_pred))