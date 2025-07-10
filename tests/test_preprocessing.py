import pytest
import pandas as pd
import numpy as np
from preprocessing import Preprocessor, PreprocessingPipeline
from preprocessing.transformers import (
    NumericScaler, CategoricalEncoder, MissingValueHandler,
    OutlierHandler, FeatureSelector
)


class TestMissingValueHandler:
    """欠損値処理のロジック保証"""
    
    def test_numeric_mean_imputation(self, mini_regression_df):
        """数値列の平均値補完"""
        handler = MissingValueHandler(numeric_strategy='mean')
        df_imputed = handler.fit_transform(mini_regression_df)
        
        # 欠損値が0になっていることを確認
        assert df_imputed['numeric_feat1'].isnull().sum() == 0
        
        # 平均値で補完されていることを確認
        original_mean = mini_regression_df['numeric_feat1'].mean()
        assert df_imputed['numeric_feat1'].mean() == pytest.approx(original_mean, rel=1e-3)
    
    def test_categorical_mode_imputation(self, mini_regression_df):
        """カテゴリ列のモード補完"""
        handler = MissingValueHandler(categorical_strategy='mode')
        df_imputed = handler.fit_transform(mini_regression_df)
        
        # 欠損値が0になっていることを確認
        assert df_imputed['categorical_feat1'].isnull().sum() == 0


class TestCategoricalEncoder:
    """カテゴリエンコーディングのロジック保証"""
    
    def test_label_encoding_preserves_categories(self, mini_regression_df):
        """ラベルエンコーディングがカテゴリを保持"""
        encoder = CategoricalEncoder(method='label')
        df_encoded = encoder.fit_transform(mini_regression_df)
        
        # エンコード後も値の種類数が保持されている
        original_unique = mini_regression_df['categorical_feat1'].dropna().nunique()
        encoded_unique = df_encoded['categorical_feat1'].nunique()
        assert encoded_unique >= original_unique  # 欠損値分が追加される可能性
    
    def test_unknown_category_handling(self, mini_regression_df):
        """未知カテゴリの処理"""
        encoder = CategoricalEncoder(method='label')
        encoder.fit(mini_regression_df)
        
        # 未知カテゴリを含むデータを作成
        test_df = mini_regression_df.copy()
        test_df.loc[0, 'categorical_feat1'] = 'UNKNOWN'
        
        df_encoded = encoder.transform(test_df)
        
        # 未知カテゴリが適切に処理されている
        assert not df_encoded['categorical_feat1'].isnull().any()


class TestNumericScaler:
    """数値スケーリングのロジック保証"""
    
    def test_standard_scaling_properties(self, mini_regression_df):
        """標準化の性質確認"""
        scaler = NumericScaler(method='standard')
        df_scaled = scaler.fit_transform(mini_regression_df)
        
        # スケール後の平均と標準偏差
        scaled_col = df_scaled['numeric_feat1'].dropna()
        assert scaled_col.mean() == pytest.approx(0, abs=1e-10)
        # ddof=1 for sample std deviation, so it's slightly different from 1.0
        assert scaled_col.std() == pytest.approx(1, abs=0.01)
    
    def test_minmax_scaling_range(self, mini_regression_df):
        """MinMaxスケーリングの範囲確認"""
        scaler = NumericScaler(method='minmax')
        df_scaled = scaler.fit_transform(mini_regression_df)
        
        scaled_col = df_scaled['numeric_feat1'].dropna()
        assert scaled_col.min() >= 0
        assert scaled_col.max() <= 1


class TestOutlierHandler:
    """外れ値処理のロジック保証"""
    
    def test_iqr_clipping(self, mini_regression_df):
        """IQRによる外れ値除去"""
        handler = OutlierHandler(method='iqr', threshold=1.5)
        df_clipped = handler.fit_transform(mini_regression_df)
        
        # クリッピング前後でデータ形状は同じ
        assert df_clipped.shape == mini_regression_df.shape
        
        # 極端な外れ値が除去されている
        original_range = mini_regression_df['numeric_feat1'].max() - mini_regression_df['numeric_feat1'].min()
        clipped_range = df_clipped['numeric_feat1'].max() - df_clipped['numeric_feat1'].min()
        assert clipped_range <= original_range


class TestPreprocessingPipeline:
    """前処理パイプラインの統合テスト"""
    
    def test_default_pipeline_runs_without_error(self, mini_regression_df):
        """デフォルトパイプラインがエラーなく実行"""
        pipeline = PreprocessingPipeline()
        df_processed = pipeline.fit_transform(mini_regression_df)
        
        # 出力が生成される
        assert df_processed is not None
        assert len(df_processed) > 0
        
        # 欠損値が処理されている
        assert df_processed.isnull().sum().sum() == 0
    
    def test_pipeline_reproducibility(self, mini_regression_df):
        """パイプラインの再現性"""
        pipeline1 = PreprocessingPipeline()
        pipeline2 = PreprocessingPipeline()
        
        df1 = pipeline1.fit_transform(mini_regression_df)
        df2 = pipeline2.fit_transform(mini_regression_df)
        
        # 同じ結果が得られる
        pd.testing.assert_frame_equal(df1, df2)


class TestPreprocessor:
    """Preprocessorクラスの統合テスト"""
    
    def test_prepare_data_splits_correctly(self, mini_regression_df):
        """データ分割が正しく動作"""
        preprocessor = Preprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(
            mini_regression_df, 'target', test_size=0.2, random_state=42
        )
        
        # 分割比率が正しい
        total_rows = len(mini_regression_df)
        assert len(X_train) == pytest.approx(total_rows * 0.8, abs=5)
        assert len(X_test) == pytest.approx(total_rows * 0.2, abs=5)
        
        # targetが除外されている
        assert 'target' not in X_train.columns
        assert 'target' not in X_test.columns
    
    def test_quick_preprocess_end_to_end(self, mini_regression_df):
        """クイック前処理のエンドツーエンド"""
        result = Preprocessor.quick_preprocess(mini_regression_df, 'target')
        
        # 必要なキーが含まれている
        required_keys = ['X_train', 'X_test', 'y_train', 'y_test', 'preprocessor']
        for key in required_keys:
            assert key in result
        
        # 前処理後のデータに欠損値がない
        assert result['X_train'].isnull().sum().sum() == 0
        assert result['X_test'].isnull().sum().sum() == 0