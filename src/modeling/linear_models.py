import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression, SGDClassifier, SGDRegressor
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from .base import BaseModel, ModelConfig


class LinearModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = self._create_model()
        self.scaler = StandardScaler()
        
    def _create_model(self):
        if self.config.target_type == 'classification':
            return LogisticRegression(**self.config.params)
        else:
            return LinearRegression(**self.config.params)
            
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None):
        
        self.feature_names = X.columns.tolist()
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        if hasattr(self.model, 'coef_'):
            self.feature_importances_ = np.abs(self.model.coef_)
            if len(self.feature_importances_.shape) > 1:
                self.feature_importances_ = self.feature_importances_[0]
                
        self.fitted = True
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        if self.config.target_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class RidgeModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = self._create_model()
        self.scaler = StandardScaler()
        
    def _create_model(self):
        if self.config.target_type == 'classification':
            return LogisticRegression(penalty='l2', **self.config.params)
        else:
            return Ridge(**self.config.params)
            
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None):
        
        self.feature_names = X.columns.tolist()
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        if hasattr(self.model, 'coef_'):
            self.feature_importances_ = np.abs(self.model.coef_)
            if len(self.feature_importances_.shape) > 1:
                self.feature_importances_ = self.feature_importances_[0]
                
        self.fitted = True
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        if self.config.target_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class LassoModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = self._create_model()
        self.scaler = StandardScaler()
        
    def _create_model(self):
        if self.config.target_type == 'classification':
            return LogisticRegression(penalty='l1', solver='liblinear', **self.config.params)
        else:
            return Lasso(**self.config.params)
            
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None):
        
        self.feature_names = X.columns.tolist()
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        if hasattr(self.model, 'coef_'):
            self.feature_importances_ = np.abs(self.model.coef_)
            if len(self.feature_importances_.shape) > 1:
                self.feature_importances_ = self.feature_importances_[0]
                
        self.fitted = True
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        if self.config.target_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class ElasticNetModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = self._create_model()
        self.scaler = StandardScaler()
        
    def _create_model(self):
        if self.config.target_type == 'classification':
            return LogisticRegression(penalty='elasticnet', solver='saga', **self.config.params)
        else:
            return ElasticNet(**self.config.params)
            
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None):
        
        self.feature_names = X.columns.tolist()
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        if hasattr(self.model, 'coef_'):
            self.feature_importances_ = np.abs(self.model.coef_)
            if len(self.feature_importances_.shape) > 1:
                self.feature_importances_ = self.feature_importances_[0]
                
        self.fitted = True
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        if self.config.target_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class LogisticModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config.target_type = 'classification'
        self.model = self._create_model()
        self.scaler = StandardScaler()
        
    def _create_model(self):
        return LogisticRegression(**self.config.params)
        
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None):
        
        self.feature_names = X.columns.tolist()
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        if hasattr(self.model, 'coef_'):
            self.feature_importances_ = np.abs(self.model.coef_[0])
                
        self.fitted = True
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


def create_linear_config(target_type: str = 'regression', **params) -> ModelConfig:
    default_params = {
        'random_state': 42,
        'n_jobs': -1
    }
    
    if target_type == 'classification':
        default_params.update({
            'max_iter': 1000,
            'solver': 'lbfgs'
        })
    else:
        default_params.update({
            'fit_intercept': True
        })
    
    default_params.update(params)
    
    return ModelConfig(
        name='linear',
        model_type='linear',
        params=default_params,
        target_type=target_type
    )


def create_ridge_config(target_type: str = 'regression', **params) -> ModelConfig:
    default_params = {
        'alpha': 1.0,
        'random_state': 42,
        'max_iter': 1000
    }
    
    if target_type == 'classification':
        default_params.update({
            'solver': 'lbfgs',
            'penalty': 'l2',
            'C': 1.0
        })
    else:
        default_params.update({
            'solver': 'auto',
            'fit_intercept': True
        })
    
    default_params.update(params)
    
    return ModelConfig(
        name='ridge',
        model_type='ridge',
        params=default_params,
        target_type=target_type
    )


def create_lasso_config(target_type: str = 'regression', **params) -> ModelConfig:
    default_params = {
        'alpha': 1.0,
        'random_state': 42,
        'max_iter': 1000
    }
    
    if target_type == 'classification':
        default_params.update({
            'solver': 'liblinear',
            'penalty': 'l1',
            'C': 1.0
        })
    else:
        default_params.update({
            'fit_intercept': True
        })
    
    default_params.update(params)
    
    return ModelConfig(
        name='lasso',
        model_type='lasso',
        params=default_params,
        target_type=target_type
    )