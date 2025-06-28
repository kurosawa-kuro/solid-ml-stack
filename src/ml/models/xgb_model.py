import sys
import os

# srcディレクトリをPYTHONPATHに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Any, Union, Optional
from .base_model import BaseModel, ModelConfig
from utils.config import XGBoostConfig
from utils.base import set_seed


class XGBoostModel(BaseModel):
    """XGBoostモデルクラス"""
    
    def __init__(self, config: Optional[ModelConfig] = None, xgb_config: Optional[XGBoostConfig] = None):
        super().__init__(config)
        self.xgb_config = xgb_config or XGBoostConfig()
    
    def get_model_name(self) -> str:
        return "XGBoost"
    
    def _train(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Any:
        """XGBoostモデルを学習"""
        set_seed(self.config.seed)
        
        # --- データを確実に数値に変換 ---
        if isinstance(X, pd.DataFrame):
            X = X.copy()  # 元データを変更しないようコピー
            # カテゴリ変数をlabel encoding
            for col in X.select_dtypes(include=['object', 'category']).columns:
                X[col] = X[col].astype('category').cat.codes
            # 全ての列をfloat型に変換
            X = X.astype(float)
        elif isinstance(X, np.ndarray):
            # numpy配列の場合、文字列が含まれている可能性があるため
            # 一旦pandas DataFrameに変換してから処理
            X_df = pd.DataFrame(X)
            # カテゴリ変数をlabel encoding
            for col in X_df.select_dtypes(include=['object', 'category']).columns:
                X_df[col] = X_df[col].astype('category').cat.codes
            # 全ての列をfloat型に変換
            X = X_df.astype(float).values
        
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
        # --- データを確実に数値に変換 ---
        if isinstance(X, pd.DataFrame):
            X = X.copy()  # 元データを変更しないようコピー
            # カテゴリ変数をlabel encoding
            for col in X.select_dtypes(include=['object', 'category']).columns:
                X[col] = X[col].astype('category').cat.codes
            # 全ての列をfloat型に変換
            X = X.astype(float)
        elif isinstance(X, np.ndarray):
            # numpy配列の場合、文字列が含まれている可能性があるため
            # 一旦pandas DataFrameに変換してから処理
            X_df = pd.DataFrame(X)
            # カテゴリ変数をlabel encoding
            for col in X_df.select_dtypes(include=['object', 'category']).columns:
                X_df[col] = X_df[col].astype('category').cat.codes
            # 全ての列をfloat型に変換
            X = X_df.astype(float).values
        
        dtest = xgb.DMatrix(X)
        return model.predict(dtest)
    
    def get_feature_importance(self) -> dict:
        """特徴量重要度を取得"""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model is not fitted")
        
        importance = self.model.get_score(importance_type='gain')
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)) 