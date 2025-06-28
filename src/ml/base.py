import duckdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(db_path: str, table: str = "gold_house_features", target: str = "price"):
    """Load features and target from DuckDB."""
    con = duckdb.connect(db_path)
    df = con.execute(f"SELECT * FROM {table}").df()
    
    # カテゴリカル変数を数値に変換
    categorical_columns = ['location', 'condition']
    label_encoders = {}
    
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def set_seed(seed: int = 42):
    np.random.seed(seed)


def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """R-squared (Coefficient of Determination)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) 