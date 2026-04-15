import pandas as pd
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df

def get_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include='all').T.fillna("").round(4)

def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] == 0:
        return pd.DataFrame()
    return numeric.corr().round(3)

def get_target_distribution(df: pd.DataFrame, target: str) -> dict:
    return df[target].value_counts().to_dict()