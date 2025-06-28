import csv
from collections import defaultdict
from pathlib import Path

# プロジェクトルートを取得
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_FILE = ARTIFACTS_DIR / "ml_results.csv"

# 最新の各モデルの結果を取得
def get_latest_results():
    latest = {}
    
    if not RESULTS_FILE.exists():
        return latest
    
    with open(RESULTS_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            rmse = float(row["rmse"])
            timestamp = row["timestamp"]
            
            # MAEとR²を取得（列が存在しない場合はNone）
            mae = None
            r2 = None
            if "mae" in row and row["mae"] and row["mae"].strip():
                try:
                    mae = float(row["mae"])
                except ValueError:
                    mae = None
            
            if "r2" in row and row["r2"] and row["r2"].strip():
                try:
                    r2 = float(row["r2"])
                except ValueError:
                    r2 = None
            
            # 最新のものだけ残す
            if model not in latest or timestamp > latest[model]["timestamp"]:
                latest[model] = {
                    "rmse": rmse, 
                    "mae": mae, 
                    "r2": r2,
                    "timestamp": timestamp
                }
    return latest


def main():
    try:
        latest = get_latest_results()
    except FileNotFoundError:
        print("No results found. Please run ML targets first.")
        return
    if not latest:
        print("No results found.")
        return
    
    print("\n" + "="*80)
    print("ML MODEL PERFORMANCE REPORT")
    print("="*80)
    
    # ヘッダー
    print(f"{'Model':<12} | {'RMSE':<12} | {'MAE':<12} | {'R²':<8} | {'Timestamp'}")
    print("-" * 80)
    
    # 各モデルの結果を表示
    for model, info in latest.items():
        mae_str = f"{info['mae']:.4f}" if info['mae'] is not None else "N/A"
        r2_str = f"{info['r2']:.4f}" if info['r2'] is not None else "N/A"
        print(f"{model:<12} | {info['rmse']:<12.4f} | {mae_str:<12} | {r2_str:<8} | {info['timestamp']}")
    
    # 最良モデル（RMSE最小）
    best_rmse = min(latest.items(), key=lambda x: x[1]["rmse"])
    print(f"\n🏆 BEST MODEL (RMSE): {best_rmse[0]} (RMSE: {best_rmse[1]['rmse']:.4f})")
    
    # MAEが利用可能な場合、最良モデル（MAE最小）も表示
    models_with_mae = {k: v for k, v in latest.items() if v['mae'] is not None}
    if models_with_mae:
        best_mae = min(models_with_mae.items(), key=lambda x: x[1]["mae"])
        print(f"🥇 BEST MODEL (MAE):  {best_mae[0]} (MAE: {best_mae[1]['mae']:.4f})")
    
    # R²が利用可能な場合、最良モデル（R²最大）も表示
    models_with_r2 = {k: v for k, v in latest.items() if v['r2'] is not None}
    if models_with_r2:
        best_r2 = max(models_with_r2.items(), key=lambda x: x[1]["r2"])
        print(f"🎯 BEST MODEL (R²):   {best_r2[0]} (R²: {best_r2[1]['r2']:.4f})")
    
    print("="*80)

if __name__ == "__main__":
    main() 