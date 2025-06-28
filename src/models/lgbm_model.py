import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Any, Union, Optional
from base_model import BaseModel, ModelConfig
from config import LightGBMConfig


class LightGBMModel(BaseModel):
    """LightGBMモデルクラス"""
    
    def __init__(self, config: Optional[ModelConfig] = None, lgbm_config: Optional[LightGBMConfig] = None):
        super().__init__(config)
        self.lgbm_config = lgbm_config or LightGBMConfig()
    
    def get_model_name(self) -> str:
        return "LightGBM"
    
    def _train(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Any:
        """LightGBMモデルを学習"""
        from base import set_seed
        set_seed(self.config.seed)
        
        # Dataset作成
        dtrain = lgb.Dataset(X, label=y)
        
        # パラメータ設定
        params = {
            "objective": self.lgbm_config.objective,
            "seed": self.config.seed,
            "max_depth": self.lgbm_config.max_depth,
            "learning_rate": self.lgbm_config.learning_rate,
            "subsample": self.lgbm_config.subsample,
            "colsample_bytree": self.lgbm_config.colsample_bytree,
            "verbosity": self.lgbm_config.verbosity,
            "num_leaves": 31,
            "min_child_samples": 20,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0
        }
        
        # モデル学習
        model = lgb.train(params, dtrain, num_boost_round=self.lgbm_config.num_boost_round)
        return model
    
    def _predict(self, model: Any, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """LightGBMモデルで予測"""
        return model.predict(X)
    
    def get_feature_importance(self, importance_type: str = "gain") -> dict:
        """特徴量重要度を取得"""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model is not fitted")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        feature_names = self.model.feature_name()
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        importance_dict = dict(zip(feature_names, importance))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def get_feature_importance_split(self) -> dict:
        """分割ベースの特徴量重要度を取得"""
        return self.get_feature_importance(importance_type="split")
    
    def get_feature_importance_gain(self) -> dict:
        """ゲインベースの特徴量重要度を取得"""
        return self.get_feature_importance(importance_type="gain") 