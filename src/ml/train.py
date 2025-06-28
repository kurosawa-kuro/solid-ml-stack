import argparse
import numpy as np
from base import load_data
from xgb_runner import fit_predict_xgb
from cat_runner import fit_predict_cat
from lgbm_runner import fit_predict_lgbm
from ensemble import fit_predict_ensemble
from stack import fit_predict_stack
import csv
from datetime import datetime
import os

RESULTS_FILE = "ml_results.csv"


def save_result(model_name, rmse):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(RESULTS_FILE)
    with open(RESULTS_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "model", "rmse"])
        writer.writerow([now, model_name, rmse])


def main():
    parser = argparse.ArgumentParser(description="Simple ML Trainer")
    parser.add_argument("--db", type=str, default="../../data/dwh/solid_ml.duckdb", help="DuckDB path")
    parser.add_argument("--table", type=str, default="gold_house_features", help="Table name")
    parser.add_argument("--target", type=str, default="price", help="Target column")
    parser.add_argument("--model", type=str, default="xgb", choices=["xgb", "cat", "lgbm", "ensemble", "stack"], help="Model type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    X, y = load_data(args.db, args.table, args.target)

    if args.model == "xgb":
        _, y_pred, score = fit_predict_xgb(X, y, args.seed)
    elif args.model == "cat":
        _, y_pred, score = fit_predict_cat(X, y, args.seed)
    elif args.model == "lgbm":
        _, y_pred, score = fit_predict_lgbm(X, y, args.seed)
    elif args.model == "ensemble":
        y_pred, score = fit_predict_ensemble(X, y, args.seed)
    elif args.model == "stack":
        _, y_pred, score = fit_predict_stack(X, y, args.seed)
    else:
        raise ValueError("Unknown model type")

    print(f"Model: {args.model}")
    print(f"RMSE: {score:.4f}")
    y_pred_arr = np.asarray(y_pred).flatten()
    print(f"Predictions (first 5): {y_pred_arr[:5]}")

    # 結果を保存
    save_result(args.model, float(score))

if __name__ == "__main__":
    main() 