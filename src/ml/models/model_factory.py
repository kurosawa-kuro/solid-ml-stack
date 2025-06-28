from typing import Dict, Type, Optional
import logging

from .base_model import BaseModel, ModelConfig
from .xgb_model import XGBoostModel
from .cat_model import CatBoostModel
from .lgbm_model import LightGBMModel
from .ensemble_model import EnsembleModel, StackingModel
from src.utils.config import XGBoostConfig, CatBoostConfig, LightGBMConfig, EnsembleConfig

class ModelFactory:
    """モデルファクトリークラス"""
    _models = {
        "xgb": XGBoostModel,
        "cat": CatBoostModel,
        "lgbm": LightGBMModel,
        "ensemble": EnsembleModel,
        "stacking": EnsembleModel,  # stackingはensembleの別名
    }

    @classmethod
    def has_model(cls, name):
        """指定されたモデル名が利用可能かチェック"""
        return name in cls._models

    @classmethod
    def get_available_models(cls):
        """利用可能なモデル一覧を取得"""
        return list(cls._models.keys())

    @classmethod
    def create_model(cls, name):
        """指定されたモデル名でモデルインスタンスを作成"""
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}")
        return cls._models[name]()

# グローバルインスタンス
model_factory = ModelFactory 