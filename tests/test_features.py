import pytest
import pandas as pd
import numpy as np
from features.engineering import (
    FeatureEngineeringPipeline,
    NumericFeatureEngineer, CategoricalFeatureEngineer,
    InteractionFeatureEngineer, DatetimeFeatureEngineer
)


class TestNumericFeatureEngineer:
    """数値特徴量生成のロジック保証"""
    
    def test_log_transformation_safety(self, mini_regression_df):
        """log変換が安全に実行される"""
        engineer = NumericFeatureEngineer(transformations=['log'])
        df_features = engineer.fit_transform(mini_regression_df)
        
        # log変換特徴量が生成される
        log_cols = [col for col in df_features.columns if 'log' in col]
        assert len(log_cols) > 0
        
        # 無限大やNaNがない
        for col in log_cols:
            assert not np.isinf(df_features[col]).any()
            assert not np.isnan(df_features[col]).any()
    
    def test_sqrt_transformation(self, mini_regression_df):
        """sqrt変換の動作確認"""
        engineer = NumericFeatureEngineer(transformations=['sqrt'])
        df_features = engineer.fit_transform(mini_regression_df)
        
        sqrt_cols = [col for col in df_features.columns if 'sqrt' in col]
        assert len(sqrt_cols) > 0
        
        # 全て正の値
        for col in sqrt_cols:
            assert (df_features[col] >= 0).all()
    
    def test_feature_names_generation(self, mini_regression_df):
        """特徴量名の生成パターン確認"""
        engineer = NumericFeatureEngineer(transformations=['log', 'sqrt', 'square'])
        engineer.fit(mini_regression_df)
        
        feature_names = engineer.get_feature_names_out()
        
        # 期待される接尾辞が含まれている
        expected_suffixes = ['_log', '_sqrt', '_square']
        for suffix in expected_suffixes:
            assert any(suffix in name for name in feature_names)


class TestCategoricalFeatureEngineer:
    """カテゴリ特徴量生成のロジック保証"""
    
    def test_count_encoding_generation(self, mini_regression_df):
        """カウントエンコーディング特徴量の生成"""
        engineer = CategoricalFeatureEngineer()
        df_features = engineer.fit_transform(mini_regression_df)
        
        # count, frequency, is_rare特徴量が生成される
        count_cols = [col for col in df_features.columns if '_count' in col]
        freq_cols = [col for col in df_features.columns if '_frequency' in col]
        rare_cols = [col for col in df_features.columns if '_is_rare' in col]
        
        assert len(count_cols) > 0
        assert len(freq_cols) > 0
        assert len(rare_cols) > 0
    
    def test_frequency_values_range(self, mini_regression_df):
        """頻度特徴量の値範囲確認"""
        engineer = CategoricalFeatureEngineer()
        df_features = engineer.fit_transform(mini_regression_df)
        
        freq_cols = [col for col in df_features.columns if '_frequency' in col]
        
        for col in freq_cols:
            # 頻度は0-1の範囲
            assert (df_features[col] >= 0).all()
            assert (df_features[col] <= 1).all()


class TestInteractionFeatureEngineer:
    """交互作用特徴量生成のロジック保証"""
    
    def test_interaction_feature_creation(self, mini_regression_df):
        """交互作用特徴量の生成"""
        engineer = InteractionFeatureEngineer(max_interactions=10)
        df_features = engineer.fit_transform(mini_regression_df)
        
        # 掛け算、割り算、足し算特徴量が生成される
        multiply_cols = [col for col in df_features.columns if '_multiply_' in col]
        divide_cols = [col for col in df_features.columns if '_divide_' in col]
        add_cols = [col for col in df_features.columns if '_add_' in col]
        
        assert len(multiply_cols) > 0
        assert len(divide_cols) > 0
        assert len(add_cols) > 0
    
    def test_divide_by_zero_safety(self, mini_regression_df):
        """ゼロ除算の安全性確認"""
        # ゼロを含むデータを作成
        df_with_zero = mini_regression_df.copy()
        df_with_zero.loc[0, 'numeric_feat2'] = 0
        
        engineer = InteractionFeatureEngineer(max_interactions=5)
        df_features = engineer.fit_transform(df_with_zero)
        
        # 無限大がない
        divide_cols = [col for col in df_features.columns if '_divide_' in col]
        for col in divide_cols:
            assert not np.isinf(df_features[col]).any()


class TestDatetimeFeatureEngineer:
    """日時特徴量生成のロジック保証"""
    
    def test_datetime_feature_extraction(self, mini_regression_df):
        """日時特徴量の抽出"""
        engineer = DatetimeFeatureEngineer()
        df_features = engineer.fit_transform(mini_regression_df)
        
        # 年、月、日、曜日特徴量が生成される
        expected_features = ['year', 'month', 'day', 'dayofweek']
        
        for feature in expected_features:
            feature_cols = [col for col in df_features.columns if feature in col]
            assert len(feature_cols) > 0
    
    def test_weekend_flag_generation(self, mini_regression_df):
        """週末フラグの生成"""
        engineer = DatetimeFeatureEngineer(features=['is_weekend'])
        df_features = engineer.fit_transform(mini_regression_df)
        
        weekend_cols = [col for col in df_features.columns if 'is_weekend' in col]
        assert len(weekend_cols) > 0
        
        # 0または1の値
        for col in weekend_cols:
            assert set(df_features[col].unique()).issubset({0, 1})


class TestFeatureEngineeringPipeline:
    """特徴量エンジニアリングパイプラインの統合テスト"""
    
    def test_pipeline_feature_generation(self, mini_regression_df):
        """パイプラインによる特徴量生成"""
        pipeline = FeatureEngineeringPipeline()
        pipeline.add_numeric_features(['log', 'sqrt'])
        pipeline.add_categorical_features()
        pipeline.add_datetime_features()
        
        df_features = pipeline.fit_transform(mini_regression_df)
        
        # 元のデータより特徴量が増加
        assert df_features.shape[1] > mini_regression_df.shape[1]
        
        # さまざまな種類の特徴量が生成されている
        feature_types = []
        if any('_log' in col for col in df_features.columns):
            feature_types.append('numeric')
        if any('_count' in col for col in df_features.columns):
            feature_types.append('categorical')
        if any('_year' in col for col in df_features.columns):
            feature_types.append('datetime')
        
        assert len(feature_types) >= 2  # 複数種類の特徴量が生成
    
    def test_pipeline_reproducibility(self, mini_regression_df):
        """パイプラインの再現性"""
        pipeline1 = FeatureEngineeringPipeline()
        pipeline1.add_numeric_features(['log'])
        pipeline1.add_categorical_features()
        
        pipeline2 = FeatureEngineeringPipeline()
        pipeline2.add_numeric_features(['log'])
        pipeline2.add_categorical_features()
        
        df1 = pipeline1.fit_transform(mini_regression_df)
        df2 = pipeline2.fit_transform(mini_regression_df)
        
        # 同じ結果が得られる
        pd.testing.assert_frame_equal(df1, df2)


# class TestAutoFeatureEngineer:
#     """自動特徴量エンジニアリングのテスト"""
#     
#     def test_auto_engineer_default_config(self, mini_regression_df):
#         """デフォルト設定での自動特徴量生成"""
#         engineer = AutoFeatureEngineer()
#         df_features = engineer.fit_transform(mini_regression_df)
#         
#         # 特徴量が生成される
#         assert df_features.shape[1] > mini_regression_df.shape[1]
#         
#         # 欠損値がない
#         assert df_features.isnull().sum().sum() == 0
#     
#     def test_auto_engineer_with_target(self, mini_regression_df):
#         """ターゲット情報を使った特徴量生成"""
#         y = mini_regression_df['target']
#         X = mini_regression_df.drop(columns=['target'])
#         
#         engineer = AutoFeatureEngineer({'target_encoding': True})
#         df_features = engineer.fit_transform(X, y)
#         
#         # target encoding特徴量が生成される可能性
#         assert df_features.shape[1] >= X.shape[1]
#     
#     def test_feature_names_consistency(self, mini_regression_df):
#         """特徴量名の一貫性"""
#         engineer = AutoFeatureEngineer()
#         df_features = engineer.fit_transform(mini_regression_df)
#         
#         if engineer.pipeline:
#             feature_names = engineer.pipeline.get_feature_names_out()
#             # 生成された特徴量名がDataFrameの列名と一致
#             assert len(feature_names) == df_features.shape[1]