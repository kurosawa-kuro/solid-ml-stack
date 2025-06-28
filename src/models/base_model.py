import numpy as np
import pandas as pd
from typing import Any, Union, Optional, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

@dataclass
class ModelConfig:
    """モデル設定クラス"""
    seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    target_column: str = "price"
    categorical_columns: List[str] = field(default_factory=lambda: ['location', 'condition'])
    
    def __post_init__(self):
        if self.categorical_columns is None:
            self.categorical_columns = ['location', 'condition']

class ModelResult:
    """モデル学習結果クラス"""
    def __init__(self, score: float, y_pred: np.ndarray):
        self.score = score
        self.y_pred = y_pred

class BaseModel(ABC):
    """モデルの基底クラス"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def get_model_name(self) -> str:
        """モデル名を返す"""
        pass
    
    @abstractmethod
    def _train(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Any:
        """モデルを学習する（抽象メソッド）"""
        pass
    
    @abstractmethod
    def _predict(self, model: Any, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """モデルで予測する（抽象メソッド）"""
        pass
    
    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> ModelResult:
        """モデルを学習して予測を実行"""
        # データ型の統一
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values  # type: ignore
        
        # モデル学習
        self.model = self._train(X, y)
        self.is_fitted = True
        
        # 予測
        y_pred = self._predict(self.model, X)
        
        # スコア計算（RMSE）
        score = np.sqrt(np.mean((y - y_pred) ** 2))
        
        return ModelResult(score, y_pred)
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """学習済みモデルで予測を実行"""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model is not fitted. Call fit_predict() first.")
        
        # データ型の統一
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # 予測
        return self._predict(self.model, X)
    
    def calculate_all_metrics(self, y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> dict:
        """全評価指標を計算"""
        if isinstance(y_true, pd.Series):
            y_true = y_true.values  # type: ignore
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values  # type: ignore
        
        # RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # MAE
        mae = np.mean(np.abs(y_true - y_pred))
        
        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def get_feature_importance(self) -> dict:
        """特徴量重要度を取得（デフォルト実装）"""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model is not fitted")
        return {} 