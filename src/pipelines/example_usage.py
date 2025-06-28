#!/usr/bin/env python3
"""
新しいMLアーキテクチャの使用例
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Any, cast

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MLモジュールのインポート
try:
    from config import MLConfig, XGBoostConfig, CatBoostConfig, LightGBMConfig
    from model_factory import ModelFactory
    from base_improved import DataLoader
    from utils import save_result, print_results_summary
except ImportError as e:
    logger.error(f"Import error: {e}")
    exit(1)


def example_basic_usage():
    """基本的な使用例"""
    print("\n" + "="*50)
    print("BASIC USAGE EXAMPLE")
    print("="*50)
    
    # テストデータ作成
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X[:, 0] ** 2 + X[:, 1] * X[:, 2] + np.random.randn(100) * 0.1
    
    feature_names = [f'feature_{i}' for i in range(5)]
    X_df = pd.DataFrame(X, columns=cast(Any, feature_names))
    y_series = pd.Series(y, name='target')
    
    # デフォルト設定でファクトリー作成
    factory = ModelFactory()
    
    # XGBoostモデルを作成して学習
    xgb_model = factory.create_model("xgb")
    result = xgb_model.fit_predict(X_df, y_series)
    
    print(f"XGBoost RMSE: {result.score:.4f}")
    print(f"Predictions shape: {result.y_pred.shape}")


def example_custom_config():
    """カスタム設定の使用例"""
    print("\n" + "="*50)
    print("CUSTOM CONFIG EXAMPLE")
    print("="*50)
    
    # カスタム設定を作成
    config = MLConfig(
        seed=123,
        xgb_config=XGBoostConfig(
            num_boost_round=200,
            max_depth=8,
            learning_rate=0.05
        ),
        cat_config=CatBoostConfig(
            iterations=150,
            depth=7,
            learning_rate=0.08
        )
    )
    
    # カスタム設定でファクトリー作成
    factory = ModelFactory(config)
    
    # テストデータ
    np.random.seed(123)
    X = np.random.randn(100, 5)
    y = X[:, 0] ** 2 + X[:, 1] * X[:, 2] + np.random.randn(100) * 0.1
    
    feature_names = [f'feature_{i}' for i in range(5)]
    X_df = pd.DataFrame(X, columns=cast(Any, feature_names))
    y_series = pd.Series(y, name='target')
    
    # 複数モデルを学習
    models = ["xgb", "cat", "lgbm"]
    results = {}
    
    for model_name in models:
        if factory.has_model(model_name):
            model = factory.create_model(model_name)
            result = model.fit_predict(X_df, y_series)
            results[model_name] = result
            print(f"{model_name:8s} RMSE: {result.score:.4f}")
    
    # 最良モデルを特定
    best_model = min(results.items(), key=lambda x: x[1].score)
    print(f"\nBest model: {best_model[0]} (RMSE: {best_model[1].score:.4f})")


def example_ensemble_models():
    """エンサンブルモデルの使用例"""
    print("\n" + "="*50)
    print("ENSEMBLE MODELS EXAMPLE")
    print("="*50)
    
    # テストデータ
    np.random.seed(42)
    X = np.random.randn(200, 8)
    y = (X[:, 0] ** 2 + X[:, 1] * X[:, 2] + 
         np.sin(X[:, 3]) + np.random.randn(200) * 0.1)
    
    feature_names = [f'feature_{i}' for i in range(8)]
    X_df = pd.DataFrame(X, columns=cast(Any, feature_names))
    y_series = pd.Series(y, name='target')
    
    factory = ModelFactory()
    
    # エンサンブルモデルを学習
    ensemble_models = ["ensemble", "stacking"]
    
    for model_name in ensemble_models:
        if factory.has_model(model_name):
            print(f"\nTraining {model_name} model...")
            model = factory.create_model(model_name)
            result = model.fit_predict(X_df, y_series)
            print(f"{model_name} RMSE: {result.score:.4f}")
            
            # エンサンブルモデルの詳細情報
            if hasattr(model, 'base_models'):
                base_models = getattr(model, 'base_models', {})
                if base_models:
                    print(f"Base models: {list(base_models.keys())}")


def example_feature_importance():
    """特徴量重要度の使用例"""
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE EXAMPLE")
    print("="*50)
    
    # テストデータ（特徴量重要度が明確なデータ）
    np.random.seed(42)
    X = np.random.randn(100, 5)
    # feature_0が最も重要
    y = 3 * X[:, 0] + 0.5 * X[:, 1] + 0.1 * X[:, 2] + np.random.randn(100) * 0.1
    
    feature_names = [f'feature_{i}' for i in range(5)]
    X_df = pd.DataFrame(X, columns=cast(Any, feature_names))
    y_series = pd.Series(y, name='target')
    
    factory = ModelFactory()
    
    # XGBoostで特徴量重要度を確認
    if factory.has_model("xgb"):
        xgb_model = factory.create_model("xgb")
        result = xgb_model.fit_predict(X_df, y_series)
        
        print(f"XGBoost RMSE: {result.score:.4f}")
        
        # 特徴量重要度を取得（XGBoostモデルの場合のみ）
        if hasattr(xgb_model, 'get_feature_importance'):
            importance = getattr(xgb_model, 'get_feature_importance')()
            print("\nFeature Importance (XGBoost):")
            for feature, score in list(importance.items())[:3]:
                print(f"  {feature}: {score:.4f}")


def example_data_loading():
    """データ読み込みの使用例"""
    print("\n" + "="*50)
    print("DATA LOADING EXAMPLE")
    print("="*50)
    
    # DataLoaderの使用例
    loader = DataLoader(categorical_columns=['location', 'condition'])
    
    # 実際のデータベースがある場合
    try:
        # X, y = loader.load_data("path/to/database.duckdb")
        # print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
        print("Data loading example (commented out - requires actual database)")
    except Exception as e:
        print(f"Data loading failed: {e}")


def main():
    """メイン関数"""
    logger.info("Starting ML architecture examples...")
    
    # 各使用例を実行
    example_basic_usage()
    example_custom_config()
    example_ensemble_models()
    example_feature_importance()
    example_data_loading()
    
    print("\n" + "="*50)
    print("EXAMPLES COMPLETED")
    print("="*50)
    logger.info("ML architecture examples completed!")


if __name__ == "__main__":
    main() 