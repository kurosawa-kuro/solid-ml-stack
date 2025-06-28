import argparse
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import logging
from datetime import datetime
import os

from .base import load_data
from .config import MLConfig, DEFAULT_CONFIG
from .model_factory import model_factory
from .utils import save_result, save_predictions

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLTrainer:
    """ML学習クラス"""
    
    def __init__(self, config: Optional[MLConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.results: Dict[str, Any] = {}
    
    def train_single_model(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """単一モデルを学習"""
        logger.info(f"Training {model_name} model...")
        
        try:
            # モデル作成
            model = model_factory.create_model(model_name)
            
            # 学習と予測
            result = model.fit_predict(X, y)
            
            # 結果保存
            self.results[model_name] = {
                'model': result.model,
                'y_pred': result.y_pred,
                'score': result.score,
                'timestamp': result.timestamp
            }
            
            logger.info(f"{model_name} training completed. RMSE: {result.score:.4f}")
            return self.results[model_name]
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            raise
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """全モデルを学習"""
        available_models = model_factory.get_available_models()
        logger.info(f"Training all available models: {available_models}")
        
        for model_name in available_models:
            try:
                self.train_single_model(model_name, X, y)
            except Exception as e:
                logger.warning(f"Failed to train {model_name}: {str(e)}")
                continue
        
        return self.results
    
    def get_best_model(self) -> Optional[str]:
        """最良モデルを取得"""
        if not self.results:
            return None
        
        best_model = min(self.results.items(), key=lambda x: x[1]['score'])
        return best_model[0]
    
    def save_results(self):
        """結果を保存"""
        for model_name, result in self.results.items():
            save_result(model_name, result['score'], self.config.results_file)
            
            if self.config.save_predictions:
                save_predictions(
                    model_name, 
                    result['y_pred'], 
                    result['timestamp'],
                    self.config.models_dir
                )


def main():
    parser = argparse.ArgumentParser(description="Refactored ML Trainer")
    parser.add_argument("--db", type=str, default="../../data/dwh/solid_ml.duckdb", help="DuckDB path")
    parser.add_argument("--table", type=str, default="gold_house_features", help="Table name")
    parser.add_argument("--target", type=str, default="price", help="Target column")
    parser.add_argument("--model", type=str, default="all", help="Model type (or 'all')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", type=str, help="Config file path")
    args = parser.parse_args()
    
    # 設定読み込み
    config = DEFAULT_CONFIG
    config.seed = args.seed
    config.target_column = args.target
    
    # データ読み込み
    logger.info("Loading data...")
    X, y = load_data(args.db, args.table, args.target)
    logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # トレーナー作成
    trainer = MLTrainer(config)
    
    # モデル学習
    if args.model == "all":
        results = trainer.train_all_models(X, y)
    else:
        result = trainer.train_single_model(args.model, X, y)
        results = {args.model: result}
    
    # 結果表示
    print("\n=== Training Results ===")
    for model_name, result in results.items():
        print(f"{model_name:10s} | RMSE: {result['score']:.4f}")
    
    # 最良モデル表示
    best_model = trainer.get_best_model()
    if best_model:
        print(f"\nBest Model: {best_model} (RMSE: {results[best_model]['score']:.4f})")
    
    # 結果保存
    trainer.save_results()
    logger.info("Results saved successfully")


if __name__ == "__main__":
    main() 