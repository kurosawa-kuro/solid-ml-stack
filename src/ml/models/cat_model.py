import sys
import os

# srcディレクトリをPYTHONPATHに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from typing import Any, Union, Optional
from .base_model import BaseModel, ModelConfig
from utils.config import CatBoostConfig
from utils.base import set_seed


class CatBoostModel(BaseModel):
    """CatBoostモデルクラス"""
    
    def __init__(self, config: Optional[ModelConfig] = None, cat_config: Optional[CatBoostConfig] = None):
        super().__init__(config)
        self.cat_config = cat_config or CatBoostConfig()
    
    def get_model_name(self) -> str:
        return "CatBoost"
    
    def _train(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Any:
        """CatBoostモデルを学習"""
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
        
        # パラメータ設定
        params = {
            "iterations": self.cat_config.iterations,
            "depth": self.cat_config.depth,
            "learning_rate": self.cat_config.learning_rate,
            "loss_function": self.cat_config.loss_function,
            "verbose": self.cat_config.verbose,
            "random_seed": self.config.seed,
            "train_dir": self.cat_config.train_dir  # CatBoost情報の出力先を指定
        }
        
        # モデル作成と学習
        model = CatBoostRegressor(**params)
        model.fit(X, y)
        
        return model
    
    def _predict(self, model: Any, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """CatBoostモデルで予測"""
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