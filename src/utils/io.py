import csv
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, Optional
import json
from pathlib import Path

# プロジェクトルートを取得
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_RESULTS_FILE = ARTIFACTS_DIR / "ml_results.csv"


def save_result(model_name: str, score: float, results_file: Optional[str] = None):
    """結果をCSVファイルに保存"""
    if results_file is None:
        results_file = str(DEFAULT_RESULTS_FILE)
        # artifactsディレクトリが存在しない場合は作成
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(results_file)
    
    with open(results_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "model", "rmse"])
        writer.writerow([now, model_name, score])


def save_predictions(model_name: str, y_pred: np.ndarray, timestamp: datetime, models_dir: str):
    """予測結果を保存"""
    os.makedirs(models_dir, exist_ok=True)
    
    # 予測結果をDataFrameに変換
    pred_df = pd.DataFrame({
        'prediction': y_pred,
        'timestamp': timestamp
    })
    
    # ファイル名生成
    filename = f"{model_name}_predictions_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(models_dir, filename)
    
    pred_df.to_csv(filepath, index=False)


def load_latest_results(results_file: Optional[str] = None) -> dict:
    """最新の結果を読み込み"""
    if results_file is None:
        results_file = str(DEFAULT_RESULTS_FILE)
    
    latest = {}
    
    if not os.path.exists(results_file):
        return latest
    
    with open(results_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            rmse = float(row["rmse"])
            timestamp = row["timestamp"]
            
            # 最新のものだけ残す
            if model not in latest or timestamp > latest[model]["timestamp"]:
                latest[model] = {"rmse": rmse, "timestamp": timestamp}
    
    return latest


def get_best_model(results_file: Optional[str] = None) -> Optional[tuple]:
    """最良モデルを取得"""
    if results_file is None:
        results_file = str(DEFAULT_RESULTS_FILE)
    
    latest = load_latest_results(results_file)
    
    if not latest:
        return None
    
    best_model = min(latest.items(), key=lambda x: x[1]["rmse"])
    return best_model


def save_model_config(config: dict, model_name: str, models_dir: str):
    """モデル設定を保存"""
    os.makedirs(models_dir, exist_ok=True)
    
    config_file = os.path.join(models_dir, f"{model_name}_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2, default=str)


def load_model_config(model_name: str, models_dir: str) -> Optional[dict]:
    """モデル設定を読み込み"""
    config_file = os.path.join(models_dir, f"{model_name}_config.json")
    
    if not os.path.exists(config_file):
        return None
    
    with open(config_file, 'r') as f:
        return json.load(f)


def validate_data(X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> bool:
    """データの妥当性をチェック"""
    if X is None or y is None:
        return False
    
    if len(X) != len(y):
        return False
    
    if len(X) == 0:
        return False
    
    return True


def format_score(score: float, precision: int = 4) -> str:
    """スコアをフォーマット"""
    return f"{score:.{precision}f}"


def print_results_summary(results: dict):
    """結果サマリーを表示"""
    print("\n=== Results Summary ===")
    print(f"{'Model':<12} {'RMSE':<10} {'Timestamp'}")
    print("-" * 40)
    
    for model_name, result in results.items():
        score = format_score(result['score'])
        timestamp = result['timestamp'].strftime('%Y-%m-%d %H:%M')
        print(f"{model_name:<12} {score:<10} {timestamp}")
    
    # 最良モデル
    if results:
        best_model = min(results.items(), key=lambda x: x[1]['score'])
        print(f"\nBest Model: {best_model[0]} (RMSE: {format_score(best_model[1]['score'])})") 