from abc import ABC, abstractmethod
from typing import Any, Tuple, Union, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """モデル設定のデータクラス"""
    seed: int = 42
    verbose: bool = False
    save_model: bool = False
    model_path: str = "models/"


@dataclass
class ModelResult:
    """モデル結果のデータクラス"""
    model_name: str
    model: Any
    y_pred: np.ndarray
    score: float
    timestamp: datetime
    config: ModelConfig


class BaseModel(ABC):
    """MLモデルの抽象基底クラス"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def _train(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Any:
        """モデルの学習を実装"""
        pass
    
    @abstractmethod
    def _predict(self, model: Any, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """予測を実装"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """モデル名を返す"""
        pass
    
    def train(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Any:
        """モデルを学習"""
        try:
            logger.info(f"Training {self.get_model_name()} model...")
            self.model = self._train(X, y)
            self.is_fitted = True
            logger.info(f"{self.get_model_name()} training completed")
            return self.model
        except Exception as e:
            logger.error(f"Error training {self.get_model_name()}: {str(e)}")
            raise
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """予測を実行"""
        if not self.is_fitted:
            raise ValueError(f"{self.get_model_name()} model is not fitted")
        
        try:
            return self._predict(self.model, X)
        except Exception as e:
            logger.error(f"Error predicting with {self.get_model_name()}: {str(e)}")
            raise
    
    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> ModelResult:
        """学習と予測を実行して結果を返す"""
        model = self.train(X, y)
        y_pred = self.predict(X)
        score = self._calculate_score(y, y_pred)
        
        return ModelResult(
            model_name=self.get_model_name(),
            model=model,
            y_pred=y_pred,
            score=score,
            timestamp=datetime.now(),
            config=self.config
        )
    
    def _calculate_score(self, y_true: Union[np.ndarray, pd.Series], y_pred: np.ndarray) -> float:
        """スコアを計算（RMSE）"""
        from base import rmse
        return rmse(y_true, y_pred)
    
    def calculate_all_metrics(self, y_true: Union[np.ndarray, pd.Series], y_pred: np.ndarray) -> dict:
        """全評価指標を計算"""
        from base import rmse, mae, r2_score
        
        return {
            'rmse': rmse(y_true, y_pred),
            'mae': mae(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        } 