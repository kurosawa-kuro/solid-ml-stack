import argparse
import numpy as np
import pandas as pd
import sys
import os

# パスを追加してモジュールをインポート可能にする
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base import load_data
from model_factory import model_factory
import csv
from datetime import datetime

RESULTS_FILE = "ml_results.csv"


def save_result(model_name, rmse, mae=None, r2=None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(RESULTS_FILE)
    
    with open(RESULTS_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "model", "rmse", "mae", "r2"])
        writer.writerow([now, model_name, rmse, mae or "", r2 or ""])


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
    
    score = result.score
    y_pred = result.y_pred

    # 全評価指標を計算
    metrics = model.calculate_all_metrics(y, y_pred)
    
    print(f"Model: {args.model}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"R²:   {metrics['r2']:.4f}")
    y_pred_arr = np.asarray(y_pred).flatten()
    print(f"Predictions (first 5): {y_pred_arr[:5]}")

    # 結果を保存
    save_result(args.model, float(metrics['rmse']), float(metrics['mae']), float(metrics['r2']))

if __name__ == "__main__":
    main() 