import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Any, Union, Optional
from base_model import BaseModel, ModelConfig
from config import XGBoostConfig


class XGBoostModel(BaseModel):
    """XGBoostモデルクラス"""
    
    def __init__(self, config: Optional[ModelConfig] = None, xgb_config: Optional[XGBoostConfig] = None):
        super().__init__(config)
        self.xgb_config = xgb_config or XGBoostConfig()
    
    def get_model_name(self) -> str:
        return "XGBoost"
    
    def _train(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Any:
        """XGBoostモデルを学習"""
        from base import set_seed
        set_seed(self.config.seed)
        
        # DMatrix作成
        dtrain = xgb.DMatrix(X, label=y)
        
        # パラメータ設定
        params = {
            "objective": self.xgb_config.objective,
            "seed": self.config.seed,
            "max_depth": self.xgb_config.max_depth,
            "learning_rate": self.xgb_config.learning_rate,
            "subsample": self.xgb_config.subsample,
            "colsample_bytree": self.xgb_config.colsample_bytree
        }
        
        # モデル学習
        model = xgb.train(params, dtrain, num_boost_round=self.xgb_config.num_boost_round)
        return model
    
    def _predict(self, model: Any, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """XGBoostモデルで予測"""
        dtest = xgb.DMatrix(X)
        return model.predict(dtest)
    
    def get_feature_importance(self) -> dict:
        """特徴量重要度を取得"""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model is not fitted")
        
        importance = self.model.get_score(importance_type='gain')
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)) 