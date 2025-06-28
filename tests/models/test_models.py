#!/usr/bin/env python3
"""
MLモデルのテストスクリプト
新しいアーキテクチャで実装されたモデルをテストします
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, cast

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MLモジュールのインポート
try:
    from config import DEFAULT_CONFIG
    from model_factory import model_factory
    from base_improved import DataLoader
    from utils import print_results_summary
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("Please ensure you're running from the correct directory")
    exit(1)


def create_test_data(n_samples: int = 1000, n_features: int = 10, seed: int = 42) -> tuple:
    """テストデータを作成"""
    np.random.seed(seed)
    
    # 特徴量
    X = np.random.randn(n_samples, n_features)
    
    # ターゲット（非線形関係）
    y = (X[:, 0] ** 2 + X[:, 1] * X[:, 2] + np.random.randn(n_samples) * 0.1)
    
    # DataFrameに変換
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=cast(Any, feature_names))
    y_series = pd.Series(y, name='target')
    
    logger.info(f"Created test data: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    return X_df, y_series


def test_single_model(model_name: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """単一モデルをテスト"""
    logger.info(f"Testing {model_name} model...")
    
    try:
        # モデル作成
        model = model_factory.create_model(model_name)
        
        # 学習と予測
        result = model.fit_predict(X, y)
        
        logger.info(f"{model_name} test completed. RMSE: {result.score:.4f}")
        
        return {
            'model_name': model_name,
            'score': result.score,
            'y_pred': result.y_pred,
            'timestamp': result.timestamp,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Error testing {model_name}: {str(e)}")
        return {
            'model_name': model_name,
            'error': str(e),
            'success': False
        }


def test_all_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, Any]]:
    """全モデルをテスト"""
    available_models = model_factory.get_available_models()
    logger.info(f"Testing all available models: {available_models}")
    
    results = {}
    
    for model_name in available_models:
        result = test_single_model(model_name, X, y)
        results[model_name] = result
    
    return results


def test_ensemble_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, Any]]:
    """エンサンブルモデルをテスト"""
    ensemble_models = ['ensemble', 'stacking']
    results = {}
    
    for model_name in ensemble_models:
        if model_factory.has_model(model_name):
            result = test_single_model(model_name, X, y)
            results[model_name] = result
        else:
            logger.warning(f"Model {model_name} not available")
    
    return results


def print_test_summary(results: Dict[str, Dict[str, Any]]):
    """テスト結果のサマリーを表示"""
    print("\n" + "="*60)
    print("ML MODEL TEST RESULTS")
    print("="*60)
    
    successful_models = []
    failed_models = []
    
    for model_name, result in results.items():
        if result.get('success', False):
            successful_models.append((model_name, result['score']))
        else:
            failed_models.append((model_name, result.get('error', 'Unknown error')))
    
    # 成功したモデル
    if successful_models:
        print("\n✅ SUCCESSFUL MODELS:")
        print("-" * 40)
        for model_name, score in sorted(successful_models, key=lambda x: x[1]):
            print(f"{model_name:15s} | RMSE: {score:.4f}")
        
        # 最良モデル
        best_model = min(successful_models, key=lambda x: x[1])
        print(f"\n🏆 BEST MODEL: {best_model[0]} (RMSE: {best_model[1]:.4f})")
    
    # 失敗したモデル
    if failed_models:
        print("\n❌ FAILED MODELS:")
        print("-" * 40)
        for model_name, error in failed_models:
            print(f"{model_name:15s} | Error: {error}")
    
    print(f"\n📊 SUMMARY: {len(successful_models)}/{len(results)} models successful")


def main():
    """メイン関数"""
    logger.info("Starting ML model tests...")
    
    # テストデータ作成
    X, y = create_test_data(n_samples=1000, n_features=10)
    
    # 基本モデルのテスト
    logger.info("Testing basic models...")
    basic_results = test_all_models(X, y)
    
    # エンサンブルモデルのテスト
    logger.info("Testing ensemble models...")
    ensemble_results = test_ensemble_models(X, y)
    
    # 全結果を結合
    all_results = {**basic_results, **ensemble_results}
    
    # 結果表示
    print_test_summary(all_results)
    
    # 利用可能なモデル一覧
    available_models = model_factory.get_available_models()
    print(f"\n📋 Available models: {', '.join(available_models)}")
    
    logger.info("ML model tests completed!")


if __name__ == "__main__":
    main() 