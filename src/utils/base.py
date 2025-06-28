import os
import random
import numpy as np
import duckdb
import pandas as pd

try:
    import torch  # type: ignore
except ImportError:
    torch = None

def set_seed(seed: int = 42):
    """乱数シードを一括設定（numpy, random, os, torch対応）"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_data(db_path, table, target):
    """DuckDBからデータを読み込み、特徴量とターゲットに分割"""
    con = duckdb.connect(db_path)
    df = con.execute(f"SELECT * FROM {table}").df()
    con.close()
    
    X = df.drop(columns=[target])
    y = df[target]
    return X, y 