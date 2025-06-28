from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import os


@dataclass
class XGBoostConfig:
    """XGBoost設定"""
    objective: str = "reg:squarederror"
    num_boost_round: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8


@dataclass
class CatBoostConfig:
    """CatBoost設定"""
    iterations: int = 100
    depth: int = 6
    learning_rate: float = 0.1
    loss_function: str = "RMSE"
    verbose: bool = False


@dataclass
class LightGBMConfig:
    """LightGBM設定"""
    objective: str = "regression"
    num_boost_round: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    verbosity: int = -1


@dataclass
class EnsembleConfig:
    """エンサンブル設定"""
    weights: Optional[Dict[str, float]] = None
    method: str = "average"  # "average", "weighted", "stacking"


@dataclass
class MLConfig:
    """ML全般設定"""
    # 基本設定
    seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    
    # データ設定
    categorical_columns: list = field(default_factory=lambda: ['location', 'condition'])
    target_column: str = "price"
    
    # モデル設定
    xgb_config: XGBoostConfig = field(default_factory=XGBoostConfig)
    cat_config: CatBoostConfig = field(default_factory=CatBoostConfig)
    lgbm_config: LightGBMConfig = field(default_factory=LightGBMConfig)
    ensemble_config: EnsembleConfig = field(default_factory=EnsembleConfig)
    
    # 出力設定
    results_file: str = "ml_results.csv"
    models_dir: str = "models/"
    logs_dir: str = "logs/"
    
    # 実験設定
    experiment_name: str = "house_price_prediction"
    save_predictions: bool = True
    save_models: bool = False
    
    def __post_init__(self):
        """初期化後の処理"""
        # ディレクトリ作成
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)


# デフォルト設定インスタンス
DEFAULT_CONFIG = MLConfig()


def get_model_config(model_name: str, config: Optional[MLConfig] = None) -> Dict[str, Any]:
    """モデル名に応じた設定を取得"""
    config = config or DEFAULT_CONFIG
    
    configs = {
        "xgb": config.xgb_config.__dict__,
        "cat": config.cat_config.__dict__,
        "lgbm": config.lgbm_config.__dict__,
        "ensemble": config.ensemble_config.__dict__
    }
    
    return configs.get(model_name, {}) 