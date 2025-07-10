import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import os
import sys

# Add src and config directories to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
config_path = project_root / "config"

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(config_path) not in sys.path:
    sys.path.insert(0, str(config_path))


@pytest.fixture(autouse=True)
def seed_everything():
    """全体で再現性を確保する"""
    import random
    import numpy as np
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # XGBoost/LightGBM/CatBoost用
    try:
        import xgboost as xgb
        xgb.set_config(verbosity=0)
    except ImportError:
        pass
    
    try:
        import lightgbm as lgb
        # LightGBM doesn't have set_config, use logging instead
        import logging
        logging.getLogger('lightgbm').setLevel(logging.WARNING)
    except ImportError:
        pass


@pytest.fixture
def mini_regression_df():
    """回帰用ミニデータセット（100行）"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'id': range(1, n_samples + 1),
        'numeric_feat1': np.random.normal(0, 1, n_samples),
        'numeric_feat2': np.random.uniform(0, 10, n_samples),
        'categorical_feat1': np.random.choice(['A', 'B', 'C'], n_samples),
        'categorical_feat2': np.random.choice(['X', 'Y'], n_samples),
        'date_feat': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'target': np.random.normal(100, 20, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # 意図的に欠損値を作成
    df.loc[df.sample(5).index, 'numeric_feat1'] = np.nan
    df.loc[df.sample(3).index, 'categorical_feat1'] = np.nan
    
    # 重複行を作成
    df = pd.concat([df, df.iloc[:2]], ignore_index=True)
    
    return df


@pytest.fixture
def mini_classification_df():
    """分類用ミニデータセット（100行）"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'id': range(1, n_samples + 1),
        'numeric_feat1': np.random.normal(0, 1, n_samples),
        'numeric_feat2': np.random.uniform(0, 10, n_samples),
        'categorical_feat1': np.random.choice(['A', 'B', 'C'], n_samples),
        'categorical_feat2': np.random.choice(['X', 'Y'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def mini_test_df():
    """テスト用データセット（50行、targetなし）"""
    np.random.seed(42)
    n_samples = 50
    
    data = {
        'id': range(1, n_samples + 1),
        'numeric_feat1': np.random.normal(0, 1, n_samples),
        'numeric_feat2': np.random.uniform(0, 10, n_samples),
        'categorical_feat1': np.random.choice(['A', 'B', 'C'], n_samples),
        'categorical_feat2': np.random.choice(['X', 'Y'], n_samples),
        'date_feat': pd.date_range('2023-01-01', periods=n_samples, freq='D')
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_dir():
    """一時ディレクトリ"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_csv_files(mini_regression_df, mini_test_df, temp_dir):
    """一時CSVファイル（train/test）"""
    train_path = temp_dir / 'train.csv'
    test_path = temp_dir / 'test.csv'
    
    mini_regression_df.to_csv(train_path, index=False)
    mini_test_df.to_csv(test_path, index=False)
    
    return {
        'train': train_path,
        'test': test_path
    }


@pytest.fixture
def expected_baseline_metrics():
    """ベースライン指標（回帰劣化テスト用）"""
    return {
        'rmse_threshold': 31.0,  # これより悪くなったら失敗
        'mae_threshold': 25.0,
        'r2_threshold': -0.5     # R²がこれより低くなったら失敗
    }


@pytest.fixture
def expected_classification_metrics():
    """ベースライン指標（分類劣化テスト用）"""
    return {
        'accuracy_threshold': 0.30,  # これより悪くなったら失敗
        'auc_threshold': 0.45
    }


@pytest.fixture
def sample_predictions():
    """予測結果サンプル"""
    return {
        'regression': np.random.normal(100, 20, 50),
        'classification': np.random.uniform(0, 1, 50),
        'classification_binary': np.random.choice([0, 1], 50)
    }