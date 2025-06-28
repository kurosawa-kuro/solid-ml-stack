import csv
from collections import defaultdict

RESULTS_FILE = "ml_results.csv"

# 最新の各モデルの結果を取得
def get_latest_results():
    latest = {}
    with open(RESULTS_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            rmse = float(row["rmse"])
            timestamp = row["timestamp"]
            # 最新のものだけ残す
            if model not in latest or timestamp > latest[model]["timestamp"]:
                latest[model] = {"rmse": rmse, "timestamp": timestamp}
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
    print("\n=== Latest ML Results ===")
    for model, info in latest.items():
        print(f"{model:10s} | RMSE: {info['rmse']:.4f} | {info['timestamp']}")
    # 最も精度の良いモデル
    best_model = min(latest.items(), key=lambda x: x[1]["rmse"])
    print("\nBest Model:")
    print(f"{best_model[0]} (RMSE: {best_model[1]['rmse']:.4f})")

if __name__ == "__main__":
    main() 