import pytest
import pandas as pd
import numpy as np
from modeling import (
    ModelFactory, create_kaggle_models,
    XGBoostModel, LightGBMModel, CatBoostModel,
    create_xgboost_config, create_lightgbm_config, create_catboost_config
)
from modeling.ensemble import EnsembleModel, StackingEnsemble
from preprocessing import Preprocessor


class TestModelFactory:
    """モデルファクトリのテスト"""
    
    def test_available_models_list(self):
        """利用可能モデルの確認"""
        models = ModelFactory.get_available_models()
        
        # 主要なモデルが含まれている
        expected_models = ['xgboost', 'lightgbm', 'catboost']
        for model in expected_models:
            assert model in models
    
    def test_create_model_from_name(self):
        """名前からのモデル生成"""
        model = ModelFactory.create_model_from_name('xgboost', target_type='regression')
        
        assert model is not None
        assert model.config.model_type == 'xgboost'
        assert model.config.target_type == 'regression'
    
    def test_default_models_creation(self):
        """デフォルトモデルセットの作成"""
        models = ModelFactory.get_default_models(target_type='regression')
        
        assert len(models) >= 3  # XGB, LGB, CAT最低限
        model_types = [m.config.model_type for m in models]
        assert 'xgboost' in model_types
        assert 'lightgbm' in model_types
        assert 'catboost' in model_types


class TestTreeModels:
    """ツリーモデルの基本動作テスト"""
    
    def test_xgboost_fit_predict_regression(self, mini_regression_df):
        """XGBoost回帰の学習・予測"""
        # データ準備
        preprocessor = Preprocessor()
        result = preprocessor.quick_preprocess(mini_regression_df, 'target')
        X_train, X_test = result['X_train'], result['X_test']
        y_train, y_test = result['y_train'], result['y_test']
        
        # モデル作成・学習
        config = create_xgboost_config(target_type='regression', n_estimators=10)
        model = XGBoostModel(config)
        model.fit(X_train, y_train, X_test, y_test)
        
        # 予測
        predictions = model.predict(X_test)
        
        # 予測結果の検証
        assert predictions is not None
        assert len(predictions) == len(X_test)
        assert not np.isnan(predictions).any()
        assert not np.isinf(predictions).any()
    
    def test_lightgbm_fit_predict_regression(self, mini_regression_df):
        """LightGBM回帰の学習・予測"""
        preprocessor = Preprocessor()
        result = preprocessor.quick_preprocess(mini_regression_df, 'target')
        X_train, X_test = result['X_train'], result['X_test']
        y_train, y_test = result['y_train'], result['y_test']
        
        config = create_lightgbm_config(target_type='regression', n_estimators=10)
        model = LightGBMModel(config)
        model.fit(X_train, y_train, X_test, y_test)
        
        predictions = model.predict(X_test)
        
        assert predictions is not None
        assert len(predictions) == len(X_test)
        assert not np.isnan(predictions).any()
    
    def test_catboost_fit_predict_regression(self, mini_regression_df):
        """CatBoost回帰の学習・予測"""
        preprocessor = Preprocessor()
        result = preprocessor.quick_preprocess(mini_regression_df, 'target')
        X_train, X_test = result['X_train'], result['X_test']
        y_train, y_test = result['y_train'], result['y_test']
        
        config = create_catboost_config(target_type='regression', iterations=10)
        model = CatBoostModel(config)
        model.fit(X_train, y_train, X_test, y_test)
        
        predictions = model.predict(X_test)
        
        assert predictions is not None
        assert len(predictions) == len(X_test)
        assert not np.isnan(predictions).any()
    
    def test_classification_predict_proba(self, mini_classification_df):
        """分類での確率予測"""
        preprocessor = Preprocessor()
        result = preprocessor.quick_preprocess(mini_classification_df, 'target')
        X_train, X_test = result['X_train'], result['X_test']
        y_train, y_test = result['y_train'], result['y_test']
        
        config = create_xgboost_config(target_type='classification', n_estimators=10)
        model = XGBoostModel(config)
        model.fit(X_train, y_train, X_test, y_test)
        
        probabilities = model.predict_proba(X_test)
        
        assert probabilities is not None
        assert probabilities.shape[0] == len(X_test)
        assert probabilities.shape[1] == 2  # 二値分類
        assert np.all((probabilities >= 0) & (probabilities <= 1))


class TestModelPerformance:
    """モデル性能の回帰テスト"""
    
    def test_regression_baseline_performance(self, mini_regression_df, expected_baseline_metrics):
        """回帰性能のベースライン確認"""
        preprocessor = Preprocessor()
        result = preprocessor.quick_preprocess(mini_regression_df, 'target')
        X_train, X_test = result['X_train'], result['X_test']
        y_train, y_test = result['y_train'], result['y_test']
        
        config = create_xgboost_config(target_type='regression', n_estimators=20)
        model = XGBoostModel(config)
        model.fit(X_train, y_train, X_test, y_test)
        
        predictions = model.predict(X_test)
        
        # 性能指標計算
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # ベースライン性能を下回らない
        assert rmse <= expected_baseline_metrics['rmse_threshold']
        assert mae <= expected_baseline_metrics['mae_threshold']
        assert r2 >= expected_baseline_metrics['r2_threshold']
    
    def test_classification_baseline_performance(self, mini_classification_df, expected_classification_metrics):
        """分類性能のベースライン確認"""
        preprocessor = Preprocessor()
        result = preprocessor.quick_preprocess(mini_classification_df, 'target')
        X_train, X_test = result['X_train'], result['X_test']
        y_train, y_test = result['y_train'], result['y_test']
        
        config = create_xgboost_config(target_type='classification', n_estimators=20)
        model = XGBoostModel(config)
        model.fit(X_train, y_train, X_test, y_test)
        
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        
        # 性能指標計算
        from sklearn.metrics import accuracy_score, roc_auc_score
        accuracy = accuracy_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)
        
        # ベースライン性能を下回らない
        assert accuracy >= expected_classification_metrics['accuracy_threshold']
        assert auc >= expected_classification_metrics['auc_threshold']


class TestModelReproducibility:
    """モデルの再現性テスト"""
    
    def test_same_seed_same_results(self, mini_regression_df):
        """同じシードで同じ結果"""
        preprocessor = Preprocessor()
        result = preprocessor.quick_preprocess(mini_regression_df, 'target')
        X_train, X_test = result['X_train'], result['X_test']
        y_train, y_test = result['y_train'], result['y_test']
        
        # 同じ設定で2つのモデル
        config1 = create_xgboost_config(target_type='regression', n_estimators=10, random_state=42)
        config2 = create_xgboost_config(target_type='regression', n_estimators=10, random_state=42)
        
        model1 = XGBoostModel(config1)
        model2 = XGBoostModel(config2)
        
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)
        
        pred1 = model1.predict(X_test)
        pred2 = model2.predict(X_test)
        
        # 完全一致
        np.testing.assert_array_equal(pred1, pred2)


class TestEnsembleMethods:
    """アンサンブル手法のテスト"""
    
    def test_ensemble_average_prediction(self, mini_regression_df):
        """平均アンサンブルの予測"""
        preprocessor = Preprocessor()
        result = preprocessor.quick_preprocess(mini_regression_df, 'target')
        X_train, X_test = result['X_train'], result['X_test']
        y_train, y_test = result['y_train'], result['y_test']
        
        # 複数モデル作成
        models = []
        for model_type in ['xgboost', 'lightgbm']:
            model = ModelFactory.create_model_from_name(
                model_type, target_type='regression', n_estimators=10
            )
            model.fit(X_train, y_train)
            models.append(model)
        
        # アンサンブル
        ensemble = EnsembleModel(models, ensemble_method='average')
        ensemble.fit(X_train, y_train)
        
        predictions = ensemble.predict(X_test)
        
        assert predictions is not None
        assert len(predictions) == len(X_test)
        assert not np.isnan(predictions).any()
    
    def test_stacking_ensemble(self, mini_regression_df):
        """スタッキングアンサンブルのテスト"""
        preprocessor = Preprocessor()
        result = preprocessor.quick_preprocess(mini_regression_df, 'target')
        X_train, X_test = result['X_train'], result['X_test']
        y_train, y_test = result['y_train'], result['y_test']
        
        # ベースモデル作成（軽量化）
        base_models = []
        for model_type in ['xgboost', 'lightgbm']:
            model = ModelFactory.create_model_from_name(
                model_type, target_type='regression', n_estimators=5
            )
            base_models.append(model)
        
        # スタッキング（高速化のため2-fold）
        stacking = StackingEnsemble(base_models, cv_folds=2)
        stacking.fit(X_train, y_train)
        
        predictions = stacking.predict(X_test)
        
        assert predictions is not None
        assert len(predictions) == len(X_test)
        assert not np.isnan(predictions).any()


class TestKaggleModels:
    """Kaggle向けモデルセットのテスト"""
    
    def test_kaggle_models_creation(self):
        """Kaggle向けモデルセットの作成"""
        models = create_kaggle_models(target_type='regression')
        
        assert len(models) >= 4  # 複数バリエーション
        
        # 異なる設定のモデルが含まれる
        model_names = [m.config.name for m in models]
        assert len(set(model_names)) == len(model_names)  # 重複なし
    
    def test_kaggle_models_diversity(self):
        """Kaggle向けモデルの多様性"""
        models = create_kaggle_models(target_type='regression')
        
        # 異なるモデルタイプが含まれる
        model_types = [m.config.model_type for m in models]
        unique_types = set(model_types)
        assert len(unique_types) >= 2  # 少なくとも2種類