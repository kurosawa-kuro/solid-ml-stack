from xgb_runner import fit_predict_xgb, train_xgb, predict_xgb
from cat_runner import fit_predict_cat
from lgbm_runner import fit_predict_lgbm
from base import rmse
import numpy as np
import pandas as pd


def fit_predict_stack(X, y, seed=42):
    # 1st stage: 各モデルの予測値
    _, y_pred_xgb, _ = fit_predict_xgb(X, y, seed)
    _, y_pred_cat, _ = fit_predict_cat(X, y, seed)
    _, y_pred_lgbm, _ = fit_predict_lgbm(X, y, seed)
    
    # 2nd stage: 予測値を特徴量としてXGBoostで学習
    X_stack = pd.DataFrame({
        "xgb": y_pred_xgb,
        "cat": y_pred_cat,
        "lgbm": y_pred_lgbm
    })
    model2 = train_xgb(X_stack, y, seed)
    y_pred = predict_xgb(model2, X_stack)
    score = rmse(y, y_pred)
    return model2, y_pred, score 