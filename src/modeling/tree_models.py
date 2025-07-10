import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from .base import BaseModel, ModelConfig


class XGBoostModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = self._create_model()
        
    def _create_model(self):
        if self.config.target_type == 'classification':
            return xgb.XGBClassifier(**self.config.params)
        else:
            return xgb.XGBRegressor(**self.config.params)
            
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None):
        
        self.feature_names = X.columns.tolist()
        
        eval_set = None
        fit_params = {}
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        else:
            # Remove early_stopping_rounds if no validation set
            if hasattr(self.model, 'get_params') and 'early_stopping_rounds' in self.model.get_params():
                self.model.set_params(early_stopping_rounds=None)
            
        self.model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False
        )
        
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
            
        self.fitted = True
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        if self.config.target_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        return self.model.predict_proba(X)


class LightGBMModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = self._create_model()
        
    def _create_model(self):
        if self.config.target_type == 'classification':
            return lgb.LGBMClassifier(**self.config.params)
        else:
            return lgb.LGBMRegressor(**self.config.params)
            
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None):
        
        self.feature_names = X.columns.tolist()
        
        eval_set = None
        callbacks = [lgb.log_evaluation(0)]
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks.append(lgb.early_stopping(50))
            
        self.model.fit(
            X, y,
            eval_set=eval_set,
            callbacks=callbacks
        )
        
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
            
        self.fitted = True
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        if self.config.target_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        return self.model.predict_proba(X)


class CatBoostModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = self._create_model()
        
    def _create_model(self):
        if self.config.target_type == 'classification':
            return cb.CatBoostClassifier(**self.config.params)
        else:
            return cb.CatBoostRegressor(**self.config.params)
            
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None):
        
        self.feature_names = X.columns.tolist()
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
            
        self.model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False
        )
        
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
            
        self.fitted = True
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        if self.config.target_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        return self.model.predict_proba(X)


def create_xgboost_config(target_type: str = 'regression', **params) -> ModelConfig:
    default_params = {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 50
    }
    
    if target_type == 'classification':
        default_params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        })
    else:
        default_params.update({
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse'
        })
    
    default_params.update(params)
    
    return ModelConfig(
        name='xgboost',
        model_type='xgboost',
        params=default_params,
        target_type=target_type
    )


def create_lightgbm_config(target_type: str = 'regression', **params) -> ModelConfig:
    default_params = {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1
    }
    
    if target_type == 'classification':
        default_params.update({
            'objective': 'binary',
            'metric': 'binary_logloss'
        })
    else:
        default_params.update({
            'objective': 'regression',
            'metric': 'rmse'
        })
    
    default_params.update(params)
    
    return ModelConfig(
        name='lightgbm',
        model_type='lightgbm',
        params=default_params,
        target_type=target_type
    )


def create_catboost_config(target_type: str = 'regression', **params) -> ModelConfig:
    default_params = {
        'iterations': 1000,
        'depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'verbose': False,
        'early_stopping_rounds': 50
    }
    
    if target_type == 'classification':
        default_params.update({
            'objective': 'Logloss',
            'eval_metric': 'Logloss'
        })
    else:
        default_params.update({
            'objective': 'RMSE',
            'eval_metric': 'RMSE'
        })
    
    # Handle n_estimators parameter by converting to iterations
    if 'n_estimators' in params:
        params['iterations'] = params.pop('n_estimators')
    
    # Remove conflicting parameters
    conflicting_params = ['num_boost_round', 'num_trees']
    for param in conflicting_params:
        params.pop(param, None)
    
    default_params.update(params)
    
    return ModelConfig(
        name='catboost',
        model_type='catboost',
        params=default_params,
        target_type=target_type
    )