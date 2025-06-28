from catboost import CatBoostRegressor
from base import set_seed, rmse


def train_cat(X, y, seed=42):
    set_seed(seed)
    model = CatBoostRegressor(verbose=0, random_seed=seed)
    model.fit(X, y)
    return model


def predict_cat(model, X):
    return model.predict(X)


def fit_predict_cat(X, y, seed=42):
    model = train_cat(X, y, seed)
    y_pred = predict_cat(model, X)
    score = rmse(y, y_pred)
    return model, y_pred, score 