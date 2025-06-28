import sys
import os

# srcディレクトリとmodelsディレクトリをPYTHONPATHに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from typing import Any, Union, Optional, Dict, List
from base_model import BaseModel, ModelConfig, ModelResult
from utils.config import EnsembleConfig
import logging

logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """エンサンブルモデルクラス"""
    
    def __init__(self, config: Optional[ModelConfig] = None, ensemble_config: Optional[EnsembleConfig] = None):
        super().__init__(config)
        self.ensemble_config = ensemble_config or EnsembleConfig()
        self.base_models: Dict[str, BaseModel] = {}
        self.base_predictions: Dict[str, np.ndarray] = {}
    
    def get_model_name(self) -> str:
        return "Ensemble"
    
    def _train(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """エンサンブルモデルを学習"""
        logger.info("Training ensemble model...")
        
        # 遅延インポートで循環インポートを回避
        from model_factory import model_factory
        
        # 利用可能なモデルを取得（エンサンブルモデルを除外）
        available_models = model_factory.get_available_models()
        base_models = [name for name in available_models if name not in ['ensemble', 'stacking']]
        
        if not base_models:
            raise ValueError("No base models available for ensemble")
        
        # 各ベースモデルを学習
        for model_name in base_models:
            try:
                logger.info(f"Training base model: {model_name}")
                base_model = model_factory.create_model(model_name)
                result = base_model.fit_predict(X, y)
                
                self.base_models[model_name] = base_model
                self.base_predictions[model_name] = result.y_pred
                
                logger.info(f"{model_name} trained with RMSE: {result.score:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to train {model_name}: {str(e)}")
                continue
        
        if not self.base_models:
            raise ValueError("No base models were successfully trained")
        
        # エンサンブル設定を返す
        return {
            "base_models": self.base_models,
            "base_predictions": self.base_predictions,
            "method": self.ensemble_config.method,
            "weights": self.ensemble_config.weights
        }
    
    def _predict(self, model: Dict[str, Any], X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """エンサンブルモデルで予測"""
        base_models = model["base_models"]
        method = model["method"]
        weights = model.get("weights")
        
        predictions = []
        
        # 各ベースモデルで予測
        for model_name, base_model in base_models.items():
            try:
                pred = base_model.predict(X)
                predictions.append(pred)
                logger.debug(f"{model_name} prediction shape: {pred.shape}")
            except Exception as e:
                logger.warning(f"Failed to predict with {model_name}: {str(e)}")
                continue
        
        if not predictions:
            raise ValueError("No predictions available from base models")
        
        # 予測結果を結合
        predictions_array = np.column_stack(predictions)
        
        # エンサンブル方法に応じて予測を統合
        if method == "average":
            return np.mean(predictions_array, axis=1)
        elif method == "weighted":
            if weights is None:
                # 等重みで平均
                return np.mean(predictions_array, axis=1)
            else:
                # 重み付き平均
                weight_array = np.array([weights.get(name, 1.0) for name in base_models.keys()])
                weight_array = weight_array / weight_array.sum()  # 正規化
                return np.average(predictions_array, axis=1, weights=weight_array)
        elif method == "median":
            return np.median(predictions_array, axis=1)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def get_base_model_scores(self) -> Dict[str, float]:
        """各ベースモデルのスコアを取得"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        
        # 現在の実装ではスコアを直接取得できないため、
        # 空の辞書を返すか、必要に応じて再計算する
        logger.warning("Base model scores not available in current implementation")
        return {}
    
    def get_base_model_predictions(self) -> Dict[str, np.ndarray]:
        """各ベースモデルの予測値を取得"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        
        return self.base_predictions.copy()


class StackingModel(BaseModel):
    """スタッキングモデルクラス"""
    
    def __init__(self, config: Optional[ModelConfig] = None, meta_model_name: str = "xgb"):
        super().__init__(config)
        self.meta_model_name = meta_model_name
        self.base_models: Dict[str, BaseModel] = {}
        self.meta_model: Optional[BaseModel] = None
    
    def get_model_name(self) -> str:
        return f"Stacking({self.meta_model_name})"
    
    def _train(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """スタッキングモデルを学習"""
        logger.info("Training stacking model...")
        
        # 遅延インポートで循環インポートを回避
        from model_factory import model_factory
        
        # 利用可能なモデルを取得（エンサンブルモデルを除外）
        available_models = model_factory.get_available_models()
        base_models = [name for name in available_models if name not in ['ensemble', 'stacking']]
        
        if not base_models:
            raise ValueError("No base models available for stacking")
        
        # ベースモデルを学習
        base_predictions = []
        for model_name in base_models:
            try:
                logger.info(f"Training base model: {model_name}")
                base_model = model_factory.create_model(model_name)
                result = base_model.fit_predict(X, y)
                
                self.base_models[model_name] = base_model
                base_predictions.append(result.y_pred)
                
                logger.info(f"{model_name} trained with RMSE: {result.score:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to train {model_name}: {str(e)}")
                continue
        
        if not self.base_models:
            raise ValueError("No base models were successfully trained")
        
        # メタ特徴量を作成
        meta_features = np.column_stack(base_predictions)
        
        # メタモデルを学習
        logger.info(f"Training meta model: {self.meta_model_name}")
        self.meta_model = model_factory.create_model(self.meta_model_name)
        meta_result = self.meta_model.fit_predict(meta_features, y)  # type: ignore
        
        logger.info(f"Meta model trained with RMSE: {meta_result.score:.4f}")
        
        return {
            "base_models": self.base_models,
            "meta_model": self.meta_model,
            "meta_features": meta_features
        }
    
    def _predict(self, model: Dict[str, Any], X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """スタッキングモデルで予測"""
        base_models = model["base_models"]
        meta_model = model["meta_model"]
        
        # ベースモデルで予測
        base_predictions = []
        for model_name, base_model in base_models.items():
            try:
                pred = base_model.predict(X)
                base_predictions.append(pred)
            except Exception as e:
                logger.warning(f"Failed to predict with {model_name}: {str(e)}")
                continue
        
        if not base_predictions:
            raise ValueError("No predictions available from base models")
        
        # メタ特徴量を作成
        meta_features = np.column_stack(base_predictions)
        
        # メタモデルで予測
        return meta_model.predict(meta_features) 