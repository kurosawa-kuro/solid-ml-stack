import lightgbm as lgb
from base import set_seed, rmse


def train_lgbm(X, y, seed=42):
    set_seed(seed)
    dtrain = lgb.Dataset(X, label=y)
    params = {"objective": "regression", "seed": seed, "verbosity": -1}
    model = lgb.train(params, dtrain, num_boost_round=100)
    return model


def predict_lgbm(model, X):
    return model.predict(X)


def fit_predict_lgbm(X, y, seed=42):
    model = train_lgbm(X, y, seed)
    y_pred = predict_lgbm(model, X)
    score = rmse(y, y_pred)
    return model, y_pred, score 