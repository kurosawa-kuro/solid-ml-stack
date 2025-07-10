import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from submission import SubmissionGenerator


class TestSubmissionGenerator:
    """提出ファイル生成のテスト"""
    
    def test_basic_submission_creation(self, temp_dir, sample_predictions):
        """基本的な提出ファイル作成"""
        generator = SubmissionGenerator(str(temp_dir))
        
        predictions = sample_predictions['regression']
        test_ids = np.arange(len(predictions))
        
        filepath = generator.create_submission(
            predictions, test_ids, 'target', 'id', 'test_submission.csv'
        )
        
        # ファイルが作成される
        assert Path(filepath).exists()
        
        # 内容の確認
        df = pd.read_csv(filepath)
        assert list(df.columns) == ['id', 'target']
        assert len(df) == len(predictions)
        assert df['id'].tolist() == test_ids.tolist()
        np.testing.assert_array_almost_equal(df['target'].values, predictions, decimal=10)
    
    def test_submission_without_nulls(self, temp_dir, sample_predictions):
        """提出ファイルに欠損値がない"""
        generator = SubmissionGenerator(str(temp_dir))
        
        predictions = sample_predictions['regression']
        test_ids = np.arange(len(predictions))
        
        filepath = generator.create_submission(predictions, test_ids)
        
        df = pd.read_csv(filepath)
        assert not df.isnull().any().any()
    
    def test_submission_correct_shape(self, temp_dir, sample_predictions):
        """提出ファイルの形状が正しい"""
        generator = SubmissionGenerator(str(temp_dir))
        
        predictions = sample_predictions['regression']
        test_ids = np.arange(len(predictions))
        
        filepath = generator.create_submission(predictions, test_ids)
        
        df = pd.read_csv(filepath)
        assert df.shape == (len(predictions), 2)  # id + target
    
    def test_ensemble_submission_creation(self, temp_dir):
        """アンサンブル提出ファイルの作成"""
        generator = SubmissionGenerator(str(temp_dir))
        
        # 複数モデルの予測
        n_samples = 50
        predictions_dict = {
            'model1': np.random.normal(100, 10, n_samples),
            'model2': np.random.normal(95, 12, n_samples),
            'model3': np.random.normal(105, 8, n_samples)
        }
        
        test_ids = np.arange(n_samples)
        
        filepath = generator.create_ensemble_submission(
            predictions_dict, 
            weights=None,
            test_ids=test_ids, 
            target_column='target', 
            id_column='id', 
            filename='ensemble_test.csv'
        )
        
        # ファイルが作成される
        assert Path(filepath).exists()
        
        # 内容の確認
        df = pd.read_csv(filepath)
        assert len(df) == n_samples
        assert list(df.columns) == ['id', 'target']
        
        # アンサンブル結果が各モデルの中間値付近
        individual_means = [np.mean(preds) for preds in predictions_dict.values()]
        ensemble_mean = df['target'].mean()
        
        assert min(individual_means) <= ensemble_mean <= max(individual_means)
    
    def test_weighted_ensemble_submission(self, temp_dir):
        """重み付きアンサンブル提出ファイル"""
        generator = SubmissionGenerator(str(temp_dir))
        
        n_samples = 50
        predictions_dict = {
            'model1': np.full(n_samples, 100.0),  # 固定値で重み確認
            'model2': np.full(n_samples, 200.0)
        }
        
        weights = [0.3, 0.7]  # model2を重視
        test_ids = np.arange(n_samples)
        
        filepath = generator.create_ensemble_submission(
            predictions_dict, 
            weights=weights, 
            test_ids=test_ids, 
            filename='weighted_test.csv'
        )
        
        df = pd.read_csv(filepath)
        
        # 重み付き平均が正しく計算されている
        expected_value = 100.0 * 0.3 + 200.0 * 0.7  # = 170.0
        assert all(abs(df['target'] - expected_value) < 1e-10)
    
    def test_rank_average_submission(self, temp_dir):
        """ランク平均提出ファイル"""
        generator = SubmissionGenerator(str(temp_dir))
        
        n_samples = 10
        predictions_dict = {
            'model1': np.arange(n_samples),        # 0, 1, 2, ..., 9
            'model2': np.arange(n_samples)[::-1]   # 9, 8, 7, ..., 0
        }
        
        test_ids = np.arange(n_samples)
        
        filepath = generator.create_rank_average_submission(
            predictions_dict, test_ids, filename='rank_test.csv'
        )
        
        df = pd.read_csv(filepath)
        assert len(df) == n_samples
        
        # ランク平均は中央値付近
        rank_mean = df['target'].mean()
        assert 4 <= rank_mean <= 6  # 1-10の順位の中央値付近
    
    def test_multiple_submissions_creation(self, temp_dir):
        """複数提出ファイルの一括作成"""
        generator = SubmissionGenerator(str(temp_dir))
        
        n_samples = 50
        models_predictions = {
            'xgboost': np.random.normal(100, 10, n_samples),
            'lightgbm': np.random.normal(95, 12, n_samples),
            'catboost': np.random.normal(105, 8, n_samples)
        }
        
        test_ids = np.arange(n_samples)
        
        filepaths = generator.create_multiple_submissions(
            models_predictions, test_ids
        )
        
        # 3つのファイルが作成される
        assert len(filepaths) == 3
        
        # 全ファイルが存在
        for filepath in filepaths:
            assert Path(filepath).exists()
            
            # 内容確認
            df = pd.read_csv(filepath)
            assert len(df) == n_samples
            assert list(df.columns) == ['id', 'target']


class TestSubmissionValidation:
    """提出ファイルの検証テスト"""
    
    def test_submission_validation_success(self, temp_dir, sample_predictions):
        """正常な提出ファイルの検証"""
        generator = SubmissionGenerator(str(temp_dir))
        
        predictions = sample_predictions['regression']
        test_ids = np.arange(len(predictions))
        
        filepath = generator.create_submission(predictions, test_ids)
        
        # 検証成功
        is_valid = generator.validate_submission(
            filepath, 
            expected_rows=len(predictions),
            expected_columns=['id', 'target']
        )
        
        assert is_valid
    
    def test_submission_validation_wrong_rows(self, temp_dir, sample_predictions):
        """行数が間違っている場合の検証"""
        generator = SubmissionGenerator(str(temp_dir))
        
        predictions = sample_predictions['regression']
        test_ids = np.arange(len(predictions))
        
        filepath = generator.create_submission(predictions, test_ids)
        
        # 異なる行数で検証
        is_valid = generator.validate_submission(
            filepath, 
            expected_rows=len(predictions) + 10  # 意図的に間違った値
        )
        
        assert not is_valid
    
    def test_submission_validation_missing_columns(self, temp_dir, sample_predictions):
        """列が不足している場合の検証"""
        generator = SubmissionGenerator(str(temp_dir))
        
        predictions = sample_predictions['regression']
        test_ids = np.arange(len(predictions))
        
        filepath = generator.create_submission(predictions, test_ids)
        
        # 存在しない列を期待
        is_valid = generator.validate_submission(
            filepath,
            expected_columns=['id', 'target', 'extra_column']
        )
        
        assert not is_valid


class TestSubmissionSummary:
    """提出ファイルサマリーのテスト"""
    
    def test_submission_summary_generation(self, temp_dir, sample_predictions):
        """提出ファイルサマリーの生成"""
        generator = SubmissionGenerator(str(temp_dir))
        
        predictions = sample_predictions['regression']
        test_ids = np.arange(len(predictions))
        
        filepath = generator.create_submission(predictions, test_ids)
        
        summary = generator.get_submission_summary(filepath)
        
        # 期待されるキーが含まれている
        expected_keys = ['filename', 'shape', 'columns', 'has_nulls', 'file_size_mb']
        for key in expected_keys:
            assert key in summary
        
        # 値の確認
        assert summary['shape'] == (len(predictions), 2)
        assert summary['columns'] == ['id', 'target']
        assert summary['has_nulls'] == False
        assert summary['file_size_mb'] > 0
    
    def test_submission_target_statistics(self, temp_dir):
        """ターゲット統計の確認"""
        generator = SubmissionGenerator(str(temp_dir))
        
        # 既知の統計値を持つ予測
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        test_ids = np.arange(len(predictions))
        
        filepath = generator.create_submission(predictions, test_ids)
        summary = generator.get_submission_summary(filepath)
        
        # ターゲット統計が含まれている
        assert 'target_stats' in summary
        stats = summary['target_stats']
        
        assert stats['mean'] == pytest.approx(3.0)
        assert stats['min'] == 1.0
        assert stats['max'] == 5.0
        assert stats['unique_values'] == 5