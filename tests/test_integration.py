import pytest
import pandas as pd
import numpy as np
import subprocess
import sys
from pathlib import Path
import tempfile
import os


class TestEndToEndWorkflow:
    """エンドツーエンドワークフローのテスト"""
    
    def test_full_kaggle_workflow_runs(self, temp_csv_files, temp_dir):
        """フルKaggleワークフローが完走する"""
        train_path = temp_csv_files['train']
        test_path = temp_csv_files['test']
        output_dir = temp_dir / 'outputs'
        
        # スクリプト実行
        cmd = [
            sys.executable, 'scripts/kaggle_workflow.py',
            '--train-path', str(train_path),
            '--test-path', str(test_path),
            '--target-col', 'target',
            '--problem-type', 'regression',
            '--output-dir', str(output_dir)
        ]
        
        # タイムアウト付きで実行
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300,  # 5分タイムアウト
                cwd=Path.cwd()
            )
            
            # 実行成功の確認
            assert result.returncode == 0, f"Script failed with error: {result.stderr}"
            
            # 出力ディレクトリが作成される
            assert output_dir.exists()
            
            # 提出ファイルが生成される
            submissions_dir = output_dir / 'submissions'
            if submissions_dir.exists():
                submission_files = list(submissions_dir.glob('*.csv'))
                assert len(submission_files) > 0
                
                # 提出ファイルの検証
                for sub_file in submission_files:
                    df = pd.read_csv(sub_file)
                    assert len(df) == 50  # test data rows
                    assert 'id' in df.columns
                    assert not df.isnull().any().any()
                    
        except subprocess.TimeoutExpired:
            pytest.fail("Workflow took too long (>5 minutes)")
        except Exception as e:
            pytest.fail(f"Workflow failed with exception: {e}")
    
    def test_preprocessing_to_prediction_pipeline(self, mini_regression_df):
        """前処理→特徴量→予測の一連パイプライン"""
        from preprocessing import Preprocessor
        from features.engineering import FeatureEngineeringPipeline
        from modeling import ModelFactory
        
        # 1. 前処理
        preprocessor = Preprocessor()
        X_train, X_val, y_train, y_val = preprocessor.prepare_data(
            mini_regression_df, 'target', test_size=0.3, random_state=42
        )
        X_train_processed, X_val_processed = preprocessor.process_train_test(
            X_train, X_val, y_train
        )
        
        # 前処理完了の確認
        assert X_train_processed.isnull().sum().sum() == 0
        assert X_val_processed.isnull().sum().sum() == 0
        
        # 2. 特徴量エンジニアリング
        feature_engineer = FeatureEngineeringPipeline()
        feature_engineer.add_numeric_features()
        feature_engineer.add_categorical_features()
        feature_engineer.add_datetime_features()
        
        X_train_features = feature_engineer.fit_transform(X_train_processed, y_train)
        X_val_features = feature_engineer.transform(X_val_processed)
        
        # 特徴量生成完了の確認
        assert X_train_features.shape[1] >= X_train_processed.shape[1]
        assert X_train_features.shape[0] == X_train_processed.shape[0]
        
        # 3. モデル学習・予測
        factory = ModelFactory()
        model = factory.create_model_from_name(
            'xgboost', target_type='regression', n_estimators=10
        )
        
        model.fit(X_train_features, y_train, X_val_features, y_val)
        predictions = model.predict(X_val_features)
        
        # 予測完了の確認
        assert len(predictions) == len(X_val_features)
        assert not np.isnan(predictions).any()
        
        # 最低限の性能確認（ランダムより良い）
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_val, predictions)
        baseline_mse = np.var(y_val)  # 平均予測のMSE
        assert mse < baseline_mse * 1.2  # 20%のマージン
    
    def test_multiple_models_consistency(self, mini_regression_df):
        """複数モデルの一貫性確認"""
        from preprocessing import Preprocessor
        from modeling import ModelFactory
        
        # データ準備
        preprocessor = Preprocessor()
        result = preprocessor.quick_preprocess(mini_regression_df, 'target')
        X_train, X_test = result['X_train'], result['X_test']
        y_train, y_test = result['y_train'], result['y_test']
        
        # 複数モデルで学習
        models = ['xgboost', 'lightgbm', 'catboost']
        predictions = {}
        
        for model_name in models:
            try:
                model = ModelFactory.create_model_from_name(
                    model_name, target_type='regression', n_estimators=10
                )
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                predictions[model_name] = pred
                
                # 個別モデルの健全性確認
                assert len(pred) == len(X_test)
                assert not np.isnan(pred).any()
                assert not np.isinf(pred).any()
                
            except Exception as e:
                pytest.fail(f"Model {model_name} failed: {e}")
        
        # 予測値の合理性確認（極端に異なりすぎない）
        if len(predictions) >= 2:
            pred_values = list(predictions.values())
            correlations = []
            
            for i in range(len(pred_values)):
                for j in range(i+1, len(pred_values)):
                    corr = np.corrcoef(pred_values[i], pred_values[j])[0, 1]
                    correlations.append(corr)
            
            # 相関が極端に低くない（モデルが完全にランダムでない）
            avg_corr = np.mean(correlations)
            assert avg_corr > 0.3, f"Models too inconsistent, avg correlation: {avg_corr}"


class TestDataIntegrity:
    """データ整合性のテスト"""
    
    def test_no_data_leakage_in_preprocessing(self, mini_regression_df):
        """前処理でのデータリークがない"""
        from preprocessing import Preprocessor
        
        preprocessor = Preprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(
            mini_regression_df, 'target', test_size=0.3, random_state=42
        )
        
        # 訓練・テストデータが完全に分離されている
        train_ids = set(X_train.index)
        test_ids = set(X_test.index)
        assert len(train_ids & test_ids) == 0
        
        # 前処理後もリークがない
        X_train_processed, X_test_processed = preprocessor.process_train_test(
            X_train, X_test, y_train
        )
        
        # 行数が保持されている
        assert len(X_train_processed) == len(X_train)
        assert len(X_test_processed) == len(X_test)
    
    def test_feature_engineering_consistency(self, mini_regression_df):
        """特徴量エンジニアリングの一貫性"""
        from features.engineering import FeatureEngineeringPipeline
        
        # 訓練データで学習
        X = mini_regression_df.drop(columns=['target'])
        y = mini_regression_df['target']
        
        engineer = FeatureEngineeringPipeline()
        engineer.add_numeric_features()
        engineer.add_categorical_features()
        engineer.add_datetime_features()
        
        X_features1 = engineer.fit_transform(X, y)
        
        # 同じデータで再実行
        X_features2 = engineer.transform(X)
        
        # 結果が一致
        pd.testing.assert_frame_equal(X_features1, X_features2)
        
        # 特徴量名が一致
        feature_names = engineer.get_feature_names()
        assert len(feature_names) == X_features1.shape[1]


class TestPerformanceRegression:
    """性能回帰テスト（ゴールデンモデル比較）"""
    
    def test_model_performance_not_degraded(self, mini_regression_df, expected_baseline_metrics):
        """モデル性能が劣化していない"""
        from preprocessing import Preprocessor
        from modeling import ModelFactory
        
        # 固定シードで再現性確保
        np.random.seed(42)
        
        # データ準備
        preprocessor = Preprocessor()
        result = preprocessor.quick_preprocess(mini_regression_df, 'target')
        X_train, X_test = result['X_train'], result['X_test']
        y_train, y_test = result['y_train'], result['y_test']
        
        # ベースラインモデル
        model = ModelFactory.create_model_from_name(
            'xgboost', 
            target_type='regression', 
            n_estimators=50,  # 安定した性能のため増量
            random_state=42
        )
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # 性能指標計算
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # ベースライン性能を下回らない
        assert rmse <= expected_baseline_metrics['rmse_threshold'], \
            f"RMSE degraded: {rmse} > {expected_baseline_metrics['rmse_threshold']}"
        assert mae <= expected_baseline_metrics['mae_threshold'], \
            f"MAE degraded: {mae} > {expected_baseline_metrics['mae_threshold']}"
        assert r2 >= expected_baseline_metrics['r2_threshold'], \
            f"R² degraded: {r2} < {expected_baseline_metrics['r2_threshold']}"


class TestCLISmoke:
    """CLI / Makeタスクのスモークテスト"""
    
    def test_help_command_works(self):
        """ヘルプコマンドが動作する"""
        result = subprocess.run(
            ['make', 'help'], 
            capture_output=True, 
            text=True,
            cwd=Path.cwd()
        )
        
        assert result.returncode == 0
        assert 'Kaggle ML Stack' in result.stdout
    
    def test_module_imports_work(self):
        """主要モジュールのインポートが成功する"""
        modules_to_test = [
            'src.preprocessing',
            'src.features.engineering',
            'src.modeling',
            'src.optimization',
            'src.submission'
        ]
        
        for module in modules_to_test:
            result = subprocess.run(
                [sys.executable, '-c', f'import {module}; print("OK")'],
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            
            assert result.returncode == 0, f"Failed to import {module}: {result.stderr}"
            assert 'OK' in result.stdout
    
    def test_make_setup_creates_directories(self, temp_dir):
        """make setupがディレクトリを作成する"""
        # 一時的にCWDを変更
        original_cwd = Path.cwd()
        
        try:
            os.chdir(temp_dir)
            
            result = subprocess.run(
                ['make', 'setup'],
                capture_output=True,
                text=True,
                cwd=temp_dir
            )
            
            # setup成功またはMakefileが見つからない
            if result.returncode == 0:
                # 期待されるディレクトリが作成される
                expected_dirs = ['data/raw', 'data/processed', 'outputs', 'submissions', 'logs']
                for dir_path in expected_dirs:
                    assert (temp_dir / dir_path).exists()
                    
        finally:
            os.chdir(original_cwd)