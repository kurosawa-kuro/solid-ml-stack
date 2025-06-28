import sys
import os

# srcディレクトリとmodelsディレクトリをPYTHONPATHに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from xgb_model import XGBoostModel  # type: ignore
from cat_model import CatBoostModel  # type: ignore
from lgbm_model import LightGBMModel  # type: ignore
from ensemble_model import EnsembleModel  # type: ignore

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