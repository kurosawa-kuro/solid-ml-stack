import duckdb
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle
from pathlib import Path


def create_gold_layer_features(duckdb_path: str = "data/dwh/solid_ml.duckdb"):
    """
    Gold Layer - 特徴量エンジニアリング
    仕様書に基づいてML学習に必要な高レベル特徴量を作成
    """
    
    # DuckDB接続
    conn = duckdb.connect(duckdb_path)
    
    # Silver Layerからデータを取得（完全レコードのみ）
    query = """
    SELECT * FROM silver_house_data 
    WHERE is_complete_record = true
    """
    
    df = conn.execute(query).df()
    
    if df.empty:
        print("警告: 完全レコードが見つかりません")
        return
    
    print(f"Gold Layer処理開始: {len(df)}件のレコード")
    
    # 1. 対数変換
    df['log_price'] = np.log1p(df['price'])
    df['log_sqft'] = np.log1p(df['sqft'])
    
    # 2. 多項式特徴量（2次、3次）
    df['sqft_squared'] = df['sqft'] ** 2
    df['price_per_sqft_squared'] = df['price_per_sqft'] ** 2
    df['sqft_cubed'] = df['sqft'] ** 3
    
    # 3. 交互作用特徴量
    df['price_bedrooms_interaction'] = df['price'] * df['bedrooms']
    df['price_bathrooms_interaction'] = df['price'] * df['bathrooms']
    df['sqft_bedrooms_interaction'] = df['sqft'] * df['bedrooms']
    df['sqft_bathrooms_interaction'] = df['sqft'] * df['bathrooms']
    df['price_sqft_ratio'] = df['price'] / df['sqft']
    df['bedrooms_bathrooms_interaction'] = df['bedrooms'] * df['bathrooms']
    
    # 4. カテゴリカル特徴量（ドメイン知識ベース）
    # 築年数による分類
    df['is_new_house'] = (df['house_age'] <= 10).astype(int)
    df['is_medium_age'] = ((df['house_age'] > 10) & (df['house_age'] <= 50)).astype(int)
    df['is_old_house'] = (df['house_age'] > 50).astype(int)
    
    # 面積による分類（四分位）
    sqft_q25 = df['sqft'].quantile(0.25)
    sqft_q75 = df['sqft'].quantile(0.75)
    df['is_small_house'] = (df['sqft'] <= sqft_q25).astype(int)
    df['is_large_house'] = (df['sqft'] >= sqft_q75).astype(int)
    
    # 価格による分類（四分位）
    price_q25 = df['price'].quantile(0.25)
    price_q75 = df['price'].quantile(0.75)
    df['is_affordable'] = (df['price'] <= price_q25).astype(int)
    df['is_expensive'] = (df['price'] >= price_q75).astype(int)
    
    # 5. 位置ベース特徴量
    # 地域別平均価格
    location_avg = df.groupby('location')['price'].mean().reset_index()
    location_avg.columns = ['location', 'location_avg_price']
    df = df.merge(location_avg, on='location', how='left')
    
    # 地域平均との比較
    df['price_vs_location_avg'] = df['price'] - df['location_avg_price']
    df['price_vs_location_avg_ratio'] = df['price'] / df['location_avg_price']
    
    # 地域内価格ランク
    df['location_price_rank'] = df.groupby('location')['price'].rank(pct=True)
    
    # 6. 条件スコアの数値化
    condition_mapping = {
        'POOR': 1,
        'FAIR': 2, 
        'GOOD': 3,
        'EXCELLENT': 4
    }
    df['condition_score'] = df['condition'].replace(condition_mapping)
    
    # 7. 追加の派生特徴量
    # 価格効率性（平米単価の逆数）
    df['price_efficiency'] = df['sqft'] / df['price']
    
    # 部屋密度
    df['room_density'] = (df['bedrooms'] + df['bathrooms']) / df['sqft']
    
    # 年齢と価格の交互作用
    df['age_price_interaction'] = df['house_age'] * df['price']
    
    # 8. 品質フラグの追加
    df['is_premium_location'] = df['location'].isin(['WATERFRONT', 'DOWNTOWN']).astype(int)
    df['is_high_condition'] = (df['condition_score'] >= 3).astype(int)
    
    # 9. 複合指標
    # 総合品質スコア
    df['overall_quality_score'] = (
        df['condition_score'] * 0.4 + 
        df['is_premium_location'] * 0.3 + 
        df['is_new_house'] * 0.3
    )
    
    # 価格合理性スコア
    df['price_reasonableness_score'] = (
        (df['price_vs_location_avg_ratio'] - 1).abs() * -1 + 1
    )
    
    # 10. 最終的な品質チェック
    # 無限大やNaNの処理
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col].median())
    
    # 11. 特徴量の統計情報を保存
    feature_stats = {
        'feature_count': len(df.columns),
        'record_count': len(df),
        'numeric_features': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_features': list(df.select_dtypes(include=['object']).columns),
        'processing_timestamp': datetime.now().isoformat()
    }
    
    # 12. Gold Layerテーブルを作成
    # 既存のテーブルがあれば削除
    conn.execute("DROP TABLE IF EXISTS gold_house_features")
    
    # 新しいテーブルを作成
    conn.execute("CREATE TABLE gold_house_features AS SELECT * FROM df")
    
    # 13. 特徴量エンジニアリング用のビューも作成
    conn.execute("DROP VIEW IF EXISTS v_house_analytics")
    
    analytics_view_query = """
    CREATE VIEW v_house_analytics AS
    SELECT 
        *,
        CASE 
            WHEN overall_quality_score >= 3.5 THEN 'Premium'
            WHEN overall_quality_score >= 2.5 THEN 'Standard'
            ELSE 'Basic'
        END as quality_tier,
        CASE 
            WHEN price_reasonableness_score >= 0.8 THEN 'Good Value'
            WHEN price_reasonableness_score >= 0.6 THEN 'Fair Value'
            ELSE 'Overpriced'
        END as value_assessment
    FROM gold_house_features
    """
    
    conn.execute(analytics_view_query)
    
    # 14. アーティファクト保存用ディレクトリ作成
    artifacts_dir = Path("target/preprocessing_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # 15. 特徴量名と統計情報を保存
    feature_names = list(df.columns)
    feature_names.remove('price')  # ターゲット変数は除外
    
    artifacts = {
        'feature_names': feature_names,
        'feature_stats': feature_stats,
        'condition_mapping': condition_mapping,
        'location_mapping': location_avg.set_index('location')['location_avg_price'].to_dict(),
        'processing_metadata': {
            'source_table': 'silver_house_data',
            'target_table': 'gold_house_features',
            'total_features': len(feature_names),
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    # アーティファクトを保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    artifacts_path = artifacts_dir / f"gold_features_{timestamp}.pkl"
    
    with open(artifacts_path, 'wb') as f:
        pickle.dump(artifacts, f)
    
    # 16. 処理結果の表示
    print(f"Gold Layer処理完了:")
    print(f"  - 処理レコード数: {len(df)}")
    print(f"  - 特徴量数: {len(feature_names)}")
    print(f"  - 作成テーブル: gold_house_features")
    print(f"  - 作成ビュー: v_house_analytics")
    print(f"  - アーティファクト保存: {artifacts_path}")
    
    # 特徴量の概要を表示
    print(f"\n特徴量カテゴリ:")
    print(f"  - 基本特徴量: {len([col for col in feature_names if not any(suffix in col for suffix in ['_interaction', '_squared', '_cubed', 'log_', 'is_', 'location_', 'condition_'])])}")
    print(f"  - 対数変換: {len([col for col in feature_names if col.startswith('log_')])}")
    print(f"  - 多項式: {len([col for col in feature_names if any(suffix in col for suffix in ['_squared', '_cubed'])])}")
    print(f"  - 交互作用: {len([col for col in feature_names if '_interaction' in col])}")
    print(f"  - カテゴリカル: {len([col for col in feature_names if col.startswith('is_')])}")
    print(f"  - 位置ベース: {len([col for col in feature_names if col.startswith('location_')])}")
    print(f"  - 複合指標: {len([col for col in feature_names if col.endswith('_score')])}")
    
    conn.close()
    
    return df


def create_ml_ready_features(duckdb_path: str = "data/dwh/solid_ml.duckdb"):
    """
    ML学習用の最終特徴量セットを作成
    エンコーディングとスケーリングを含む
    """
    
    # DuckDB接続
    conn = duckdb.connect(duckdb_path)
    
    # Gold Layerからデータを取得
    query = """
    SELECT * FROM gold_house_features
    """
    
    df = conn.execute(query).df()
    
    if df.empty:
        print("警告: Gold Layerデータが見つかりません")
        return
    
    print(f"ML準備処理開始: {len(df)}件のレコード")
    
    # 1. 欠損値補完
    from sklearn.impute import SimpleImputer
    
    # 数値列の欠損補完
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    numeric_imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])
    
    # カテゴリ列の欠損補完
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])
    
    # 2. ワンホットエンコーディング
    from sklearn.preprocessing import OneHotEncoder
    
    # カテゴリカル変数のエンコーディング
    if len(categorical_columns) > 0:
        ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        encoded_features = ohe.fit_transform(df[categorical_columns])
        
        # エンコードされた特徴量名を取得
        feature_names: list[str] = []
        for i, col in enumerate(categorical_columns):
            categories = ohe.categories_[i][1:]  # drop='first'なので最初を除外
            feature_names.extend([f"{col}_{cat}" for cat in categories])
        
        # エンコードされた特徴量をDataFrameに追加
        encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df.index)  # type: ignore
        df = pd.concat([df, encoded_df], axis=1)
        
        # 元のカテゴリ列を削除
        df = df.drop(columns=list(categorical_columns))
    
    # 3. 特徴量スケーリング
    from sklearn.preprocessing import StandardScaler
    
    # ターゲット変数を除外
    target_column = 'price'
    if target_column in df.columns:
        y = df[target_column]
        X = df.drop(columns=[target_column])
    else:
        X = df
        y = None
    
    # スケーリング
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # スケールされた特徴量にサフィックスを追加
    X_scaled_df.columns = [f"{col}_scaled" for col in X_scaled_df.columns]
    
    # 元の特徴量とスケールされた特徴量を結合
    if y is not None:
        final_df = pd.concat([X, X_scaled_df, y], axis=1)
    else:
        final_df = pd.concat([X, X_scaled_df], axis=1)
    
    # 4. ML準備テーブルを作成
    conn.execute("DROP TABLE IF EXISTS ft_house_ml")
    conn.execute("CREATE TABLE ft_house_ml AS SELECT * FROM final_df")
    
    # 5. アーティファクト保存
    artifacts_dir = Path("target/preprocessing_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # エンコーダーとスケーラーを保存
    ml_artifacts = {
        'feature_names': list(X.columns),
        'target_column': target_column if y is not None else None,
        'numeric_imputer': numeric_imputer,
        'categorical_imputer': categorical_imputer if len(categorical_columns) > 0 else None,
        'onehot_encoder': ohe if len(categorical_columns) > 0 else None,
        'scaler': scaler,
        'processing_metadata': {
            'source_table': 'gold_house_features',
            'target_table': 'ft_house_ml',
            'total_features': len(X.columns),
            'scaled_features': len(X_scaled_df.columns),
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    artifacts_path = artifacts_dir / f"ml_features_{timestamp}.pkl"
    with open(artifacts_path, 'wb') as f:
        pickle.dump(ml_artifacts, f)
    
    # 6. 処理結果の表示
    print(f"ML準備処理完了:")
    print(f"  - 入力特徴量数: {len(X.columns)}")
    print(f"  - スケール済み特徴量数: {len(X_scaled_df.columns)}")
    print(f"  - 作成テーブル: ft_house_ml")
    print(f"  - アーティファクト保存: {artifacts_path}")
    
    conn.close()
    
    return final_df


if __name__ == "__main__":
    # Gold Layer処理を実行
    print("=== Gold Layer 特徴量エンジニアリング開始 ===")
    gold_df = create_gold_layer_features()
    
    print("\n=== ML準備処理開始 ===")
    ml_df = create_ml_ready_features()
    
    print("\n=== 処理完了 ===")
