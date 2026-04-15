import pandas as pd
import numpy as np

def handle_zeros(df: pd.DataFrame, columns: list, action: str):
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        if action == "delete":
            df = df[df[col] != 0]

        elif action == "impute":
            median = df[col][df[col] != 0].median()
            df[col] = df[col].replace(0, median)

    return df.reset_index(drop=True)

def detect_outliers_iqr(df: pd.DataFrame):
    numeric = df.select_dtypes(include=np.number)

    if numeric.empty:
        return []

    mask = pd.Series([False] * len(df))

    for col in numeric.columns:
        Q1 = numeric[col].quantile(0.25)
        Q3 = numeric[col].quantile(0.75)
        IQR = Q3 - Q1

        outliers = (numeric[col] < Q1 - 1.5 * IQR) | \
                   (numeric[col] > Q3 + 1.5 * IQR)

        mask = mask | outliers

    return df[mask].index

def remove_outliers(df: pd.DataFrame):
    idx = detect_outliers_iqr(df)
    return df.drop(index=idx).reset_index(drop=True)

def encode_categoricals(df: pd.DataFrame):
    cat_cols = df.select_dtypes(include="object").columns
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)