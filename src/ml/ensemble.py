from xgb_runner import fit_predict_xgb
from cat_runner import fit_predict_cat
from lgbm_runner import fit_predict_lgbm
from base import rmse
import numpy as np


def fit_predict_ensemble(X, y, seed=42):
    _, y_pred_xgb, _ = fit_predict_xgb(X, y, seed)
    _, y_pred_cat, _ = fit_predict_cat(X, y, seed)
    _, y_pred_lgbm, _ = fit_predict_lgbm(X, y, seed)
    y_pred = (y_pred_xgb + y_pred_cat + y_pred_lgbm) / 3
    score = rmse(y, y_pred)
    return y_pred, score 