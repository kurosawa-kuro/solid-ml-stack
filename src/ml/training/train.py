import argparse
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
import json
from datetime import datetime
from typing import Optional

# srcディレクトリをPYTHONPATHに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.base import load_data  # type: ignore
from ml.models.model_factory import model_factory  # type: ignore

# プロジェクトルートを取得（現在のディレクトリから相対的に取得）
current_dir = Path.cwd()
if (current_dir / "src").exists():
    PROJECT_ROOT = current_dir
else:
    # src/ml/training/から実行された場合
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
METRICS_FILE = METRICS_DIR / "metrics.json"


def save_metrics(model_name: str, rmse: float, mae: Optional[float] = None, r2: Optional[float] = None):
    """メトリクスをJSONファイルに保存"""
    # 実験用サブフォルダを作成
    timestamp = datetime.now()
    experiment_name = timestamp.strftime("%Y%m%d_%H%M%S")
    experiment_dir = ARTIFACTS_DIR / "experiments" / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # メトリクスを実験フォルダに保存
    metrics_file = experiment_dir / "metrics.json"
    
    # 既存のメトリクスを読み込み
    metrics_data = {}
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
    
    # 新しいメトリクスを追加
    if model_name not in metrics_data:
        metrics_data[model_name] = []
    
    metrics_data[model_name].append({
        "timestamp": timestamp.isoformat(),
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    })
    
    # JSONファイルに保存
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    # グローバルメトリクスファイルにも保存（履歴管理用）
    global_metrics_dir = ARTIFACTS_DIR / "metrics"
    global_metrics_dir.mkdir(parents=True, exist_ok=True)
    global_metrics_file = global_metrics_dir / "metrics.json"
    
    global_metrics_data = {}
    if global_metrics_file.exists():
        with open(global_metrics_file, 'r') as f:
            global_metrics_data = json.load(f)
    
    if model_name not in global_metrics_data:
        global_metrics_data[model_name] = []
    
    global_metrics_data[model_name].append({
        "timestamp": timestamp.isoformat(),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "experiment": experiment_name
    })
    
    with open(global_metrics_file, 'w') as f:
        json.dump(global_metrics_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Simple ML Trainer")
    parser.add_argument("--db", type=str, default="../../data/dwh/solid_ml.duckdb", help="DuckDB path")
    parser.add_argument("--table", type=str, default="gold_house_features", help="Table name")
    parser.add_argument("--target", type=str, default="price", help="Target column")
    parser.add_argument("--model", type=str, default="xgb", help="Model type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    X, y = load_data(args.db, args.table, args.target)
    
    # yをpd.Seriesに確実に変換
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    # 新しいアーキテクチャを使用
    if not model_factory.has_model(args.model):
        available_models = model_factory.get_available_models()
        raise ValueError(f"Unknown model: {args.model}. Available models: {available_models}")

    # モデルを作成して学習
    model = model_factory.create_model(args.model)
    result = model.fit_predict(X, y)
    
    # 評価指標計算
    metrics = model.calculate_all_metrics(y, result.y_pred)
    
    # 結果表示
    print(f"Model: {args.model}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"R²:   {metrics['r2']:.4f}")
    print(f"Predictions (first 5): {result.y_pred[:5]}")
    
    # 特徴量重要度を表示（XGBoost、CatBoost、LightGBMの場合）
    if args.model in ['xgb', 'cat', 'lgbm']:
        try:
            importance = model.get_feature_importance()
            if importance:
                print(f"\nTop 5 Feature Importance:")
                for i, (feature, imp) in enumerate(list(importance.items())[:5]):
                    print(f"  {i+1}. {feature}: {imp:.4f}")
        except Exception as e:
            print(f"Feature importance not available: {e}")
    
    # 結果を保存
    save_metrics(args.model, float(metrics['rmse']), float(metrics['mae']), float(metrics['r2']))

if __name__ == "__main__":
    main() 