# ❤️ Heart Disease Prediction — ML Pipeline Dashboard

An end-to-end machine learning project that predicts heart disease using clinical data and presents results through an interactive Streamlit dashboard.



## 🚀 Project Overview

This project builds a complete ML pipeline including:

- Data loading and preprocessing
- Feature selection
- Model training (Logistic Regression, Random Forest, KNN)
- Evaluation using multiple metrics
- Interactive dashboard using Streamlit



## 📊 Dataset

- Cleveland Heart Disease Dataset
- ~303 patient records
- Features include:
  - Age
  - Cholesterol
  - Blood Pressure
  - Max Heart Rate
  - Chest Pain Type
- Target variable: `target` (0 = No Disease, 1 = Disease)



## ⚙️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn, Plotly
- Streamlit
- Joblib



## 🧠 Machine Learning Models

- Logistic Regression
- Random Forest (Best performing)
- K-Nearest Neighbors



## 🔍 Features

- Exploratory Data Analysis (EDA)
- Data cleaning (zero handling, outliers)
- Feature selection:
  - Variance Threshold
  - Information Gain
- Model training with 5-Fold Cross Validation
- Performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC Curve
- Interactive UI with multiple tabs



## 📈 Results

- Random Forest achieved the best performance (~90% accuracy)
- Stable results across cross-validation folds
- Good balance between precision and recall

