import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def compute_avg_metrics(cv_results):
    return {
        "accuracy": cv_results["test_accuracy"].mean(),
        "f1": cv_results["test_f1"].mean()
    }

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    return fig

def plot_roc(model, X, y):
    y_prob = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    return fig

def get_report(y_true, y_pred):
    return pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).T