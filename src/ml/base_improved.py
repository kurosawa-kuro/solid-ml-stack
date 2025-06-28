import duckdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional, Union, List, Any
import logging

# ログ設定
logger = logging.getLogger(__name__)


class DataLoader:
    """データ読み込みクラス"""
    
    def __init__(self, categorical_columns: Optional[List[str]] = None):
        self.categorical_columns = categorical_columns or ['location', 'condition']
        self.label_encoders: Dict[str, LabelEncoder] = {}
    
    def load_data(self, db_path: str, table: str = "gold_house_features", target: str = "price") -> Tuple[pd.DataFrame, pd.Series]:
        """DuckDBからデータを読み込み"""
        try:
            logger.info(f"Loading data from {db_path}, table: {table}")
            con = duckdb.connect(db_path)
            df = con.execute(f"SELECT * FROM {table}").df()
            con.close()
            
            logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
            return self._preprocess_data(df, target)
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _preprocess_data(self, df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
        """データの前処理"""
        # カテゴリカル変数を数値に変換
        df_processed = df.copy()
        
        for col in self.categorical_columns:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"Encoded categorical column: {col}")
        
        # 特徴量とターゲットを分離
        X = df_processed.drop(columns=[target])
        y = pd.Series(df_processed[target])
        
        # データ検証
        self._validate_data(X, y)
        
        return X, y
    
    def _validate_data(self, X: pd.DataFrame, y: pd.Series):
        """データの妥当性をチェック"""
        if X.empty or y.empty:
            raise ValueError("Empty dataset")
        
        if len(X) != len(y):
            raise ValueError("X and y have different lengths")
        
        # 欠損値チェック
        missing_X = X.isnull().sum().sum()
        missing_y = y.isnull().sum()
        
        if missing_X > 0:
            logger.warning(f"Found {missing_X} missing values in features")
        
        if missing_y > 0:
            logger.warning(f"Found {missing_y} missing values in target")
        
        logger.info(f"Data validation passed: {len(X)} samples, {len(X.columns)} features")


def set_seed(seed: int = 42):
    """乱数シードを設定"""
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}")


def rmse(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """RMSEを計算"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple[Any, Any, Any, Any]:
    """データを訓練・テストに分割"""
    result = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_test, y_train, y_test = result
    
    logger.info(f"Data split: train={len(X_train)}, test={len(X_test)}")
    return X_train, X_test, y_train, y_test


def load_data(db_path: str, table: str = "gold_house_features", target: str = "price") -> Tuple[pd.DataFrame, pd.Series]:
    """後方互換性のための関数"""
    loader = DataLoader()
    return loader.load_data(db_path, table, target)


# デフォルトデータローダー
default_loader = DataLoader() 