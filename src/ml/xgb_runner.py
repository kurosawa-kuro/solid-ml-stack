import xgboost as xgb
from base import set_seed, rmse


def train_xgb(X, y, seed=42):
    set_seed(seed)
    dtrain = xgb.DMatrix(X, label=y)
    params = {"objective": "reg:squarederror", "seed": seed}
    model = xgb.train(params, dtrain, num_boost_round=100)
    return model


def predict_xgb(model, X):
    dtest = xgb.DMatrix(X)
    return model.predict(dtest)


def fit_predict_xgb(X, y, seed=42):
    model = train_xgb(X, y, seed)
    y_pred = predict_xgb(model, X)
    score = rmse(y, y_pred)
    return model, y_pred, score 