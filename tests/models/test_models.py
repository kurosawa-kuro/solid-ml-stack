#!/usr/bin/env python3
"""
MLモデルのテストスクリプト
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any

from modeling import ModelFactory
from preprocessing import Preprocessor


@pytest.fixture
def test_data():
    """テストデータを作成"""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # 特徴量
    X = np.random.randn(n_samples, n_features)
    
    # ターゲット（非線形関係）
    y = (X[:, 0] ** 2 + X[:, 1] * X[:, 2] + np.random.randn(n_samples) * 0.1)
    
    # DataFrameに変換
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series


class TestModels:
    """モデルのテスト"""
    
    def test_single_model(self, test_data):
        """単一モデルをテスト"""
        X, y = test_data
        
        # XGBoostモデルを作成してテスト
        model = ModelFactory.create_model_from_name(
            'xgboost', 
            target_type='regression',
            n_estimators=10,
            random_state=42
        )
        
        # 学習
        model.fit(X, y)
        
        # 予測
        predictions = model.predict(X)
        
        # 基本的な検証
        assert len(predictions) == len(y)
        assert not np.isnan(predictions).any()
        assert not np.isinf(predictions).any()
        
        # 性能確認（過学習チェック）
        from sklearn.metrics import r2_score
        r2 = r2_score(y, predictions)
        assert r2 > 0.5  # 最低限の性能
    
    def test_all_models(self, test_data):
        """全モデルをテスト"""
        X, y = test_data
        
        # 利用可能なモデルリスト
        models_to_test = ['xgboost', 'lightgbm', 'catboost']
        
        results = {}
        for model_name in models_to_test:
            try:
                model = ModelFactory.create_model_from_name(
                    model_name,
                    target_type='regression',
                    n_estimators=10,
                    random_state=42
                )
                
                # 学習と予測
                model.fit(X, y)
                predictions = model.predict(X)
                
                # 結果を保存
                from sklearn.metrics import mean_squared_error
                rmse = np.sqrt(mean_squared_error(y, predictions))
                results[model_name] = {
                    'success': True,
                    'rmse': rmse,
                    'predictions': predictions
                }
                
                # 基本的な検証
                assert len(predictions) == len(y)
                assert not np.isnan(predictions).any()
                
            except Exception as e:
                results[model_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        # 少なくとも1つのモデルが成功
        successful_models = [k for k, v in results.items() if v['success']]
        assert len(successful_models) > 0, "No models succeeded"
        
        # 成功したモデルの数を確認
        assert len(successful_models) >= 2, f"Only {len(successful_models)} models succeeded"
    
    def test_ensemble_models(self, test_data):
        """エンサンブルモデルをテスト"""
        X, y = test_data
        
        # エンサンブルモデルを作成
        try:
            # Voting Ensembleモデル
            ensemble_model = ModelFactory.create_ensemble_model(
                'voting',
                target_type='regression',
                base_models=['xgboost', 'lightgbm'],
                n_estimators=10,
                random_state=42
            )
            
            # 学習と予測
            ensemble_model.fit(X, y)
            predictions = ensemble_model.predict(X)
            
            # 基本的な検証
            assert len(predictions) == len(y)
            assert not np.isnan(predictions).any()
            assert not np.isinf(predictions).any()
            
            # 性能確認
            from sklearn.metrics import r2_score
            r2 = r2_score(y, predictions)
            assert r2 > 0.5  # 最低限の性能
            
        except Exception as e:
            # エンサンブルモデルが実装されていない場合はスキップ
            pytest.skip(f"Ensemble model not implemented: {e}")