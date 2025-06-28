from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from typing import Any, Union, Optional
from base_model import BaseModel, ModelConfig
from config import CatBoostConfig


class CatBoostModel(BaseModel):
    """CatBoostモデルクラス"""
    
    def __init__(self, config: Optional[ModelConfig] = None, cat_config: Optional[CatBoostConfig] = None):
        super().__init__(config)
        self.cat_config = cat_config or CatBoostConfig()
    
    def get_model_name(self) -> str:
        return "CatBoost"
    
    def _train(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Any:
        """CatBoostモデルを学習"""
        from base import set_seed
        set_seed(self.config.seed)
        
        # パラメータ設定
        params = {
            "iterations": self.cat_config.iterations,
            "depth": self.cat_config.depth,
            "learning_rate": self.cat_config.learning_rate,
            "loss_function": self.cat_config.loss_function,
            "verbose": self.cat_config.verbose,
            "random_seed": self.config.seed
        }
        
        # モデル作成と学習
        model = CatBoostRegressor(**params)
        model.fit(X, y)
        
        return model
    
    def _predict(self, model: Any, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """CatBoostモデルで予測"""
        return model.predict(X)
    
    def get_feature_importance(self) -> dict:
        """特徴量重要度を取得"""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model is not fitted")
        
        importance = self.model.get_feature_importance()
        feature_names = self.model.feature_names_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        importance_dict = dict(zip(feature_names, importance))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def get_feature_importance_type(self, importance_type: str = "LossFunctionChange") -> dict:
        """指定されたタイプの特徴量重要度を取得"""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model is not fitted")
        
        importance = self.model.get_feature_importance(type=importance_type)
        feature_names = self.model.feature_names_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        importance_dict = dict(zip(feature_names, importance))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)) 