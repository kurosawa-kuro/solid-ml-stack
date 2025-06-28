from typing import Dict, Type, Optional
from base_model import BaseModel, ModelConfig
from config import MLConfig, XGBoostConfig, CatBoostConfig, LightGBMConfig, EnsembleConfig


class ModelFactory:
    """MLモデルファクトリークラス"""
    
    def __init__(self, config: Optional[MLConfig] = None):
        self.config = config or MLConfig()
        self._models: Dict[str, Type[BaseModel]] = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """デフォルトモデルを登録"""
        try:
            from models.xgb_model import XGBoostModel
            self.register_model("xgb", XGBoostModel)
        except ImportError:
            pass
        
        try:
            from models.cat_model import CatBoostModel
            self.register_model("cat", CatBoostModel)
        except ImportError:
            pass
        
        try:
            from models.lgbm_model import LightGBMModel
            self.register_model("lgbm", LightGBMModel)
        except ImportError:
            pass
        
        try:
            from models.ensemble_model import EnsembleModel, StackingModel
            self.register_model("ensemble", EnsembleModel)
            self.register_model("stacking", StackingModel)
        except ImportError:
            pass
    
    def register_model(self, name: str, model_class: Type[BaseModel]):
        """モデルクラスを登録"""
        self._models[name] = model_class
    
    def create_model(self, model_name: str, **kwargs) -> BaseModel:
        """モデルインスタンスを作成"""
        if model_name not in self._models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_class = self._models[model_name]
        
        # 設定を取得
        model_config = ModelConfig(
            seed=self.config.seed,
            save_model=self.config.save_models,
            model_path=self.config.models_dir
        )
        
        # モデル固有の設定をkwargsに追加
        if model_name == "xgb":
            xgb_config = XGBoostConfig(**self.config.xgb_config.__dict__)
            kwargs['xgb_config'] = xgb_config
        elif model_name == "cat":
            cat_config = CatBoostConfig(**self.config.cat_config.__dict__)
            kwargs['cat_config'] = cat_config
        elif model_name == "lgbm":
            lgbm_config = LightGBMConfig(**self.config.lgbm_config.__dict__)
            kwargs['lgbm_config'] = lgbm_config
        elif model_name == "ensemble":
            ensemble_config = EnsembleConfig(**self.config.ensemble_config.__dict__)
            kwargs['ensemble_config'] = ensemble_config
        
        return model_class(config=model_config, **kwargs)
    
    def get_available_models(self) -> list:
        """利用可能なモデル一覧を取得"""
        return list(self._models.keys())
    
    def has_model(self, model_name: str) -> bool:
        """モデルが利用可能かチェック"""
        return model_name in self._models


# グローバルファクトリーインスタンス
model_factory = ModelFactory() 