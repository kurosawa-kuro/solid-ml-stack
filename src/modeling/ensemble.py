import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from .base import BaseModel, ModelConfig
import warnings
warnings.filterwarnings('ignore')


class EnsembleModel(BaseModel):
    def __init__(self, models: List[BaseModel], ensemble_method: str = 'average'):
        self.models = models
        self.ensemble_method = ensemble_method
        self.weights = None
        self.meta_model = None
        self.fitted = False
        
        config = ModelConfig(
            name=f'ensemble_{ensemble_method}',
            model_type='ensemble',
            params={'method': ensemble_method},
            target_type=models[0].config.target_type if models else 'regression'
        )
        super().__init__(config)
    
    def _create_model(self):
        return None
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None):
        
        self.feature_names = X.columns.tolist()
        
        for model in self.models:
            print(f"Training {model.config.name}...")
            model.fit(X, y, X_val, y_val)
        
        if self.ensemble_method == 'stacking':
            self._fit_stacking(X, y)
        elif self.ensemble_method == 'weighted':
            self._fit_weighted(X, y, X_val, y_val)
        
        self.fitted = True
        return self
    
    def _fit_stacking(self, X: pd.DataFrame, y: pd.Series):
        meta_features = []
        
        for model in self.models:
            if self.config.target_type == 'classification':
                pred = model.predict_proba(X)
                if len(pred.shape) > 1:
                    pred = pred[:, 1]
            else:
                pred = model.predict(X)
            meta_features.append(pred)
        
        meta_X = np.column_stack(meta_features)
        
        if self.config.target_type == 'classification':
            self.meta_model = LogisticRegression(random_state=42)
        else:
            self.meta_model = LinearRegression()
        
        self.meta_model.fit(meta_X, y)
    
    def _fit_weighted(self, X: pd.DataFrame, y: pd.Series,
                     X_val: Optional[pd.DataFrame] = None,
                     y_val: Optional[pd.Series] = None):
        
        if X_val is not None and y_val is not None:
            scores = []
            for model in self.models:
                pred = model.predict(X_val)
                if self.config.target_type == 'classification':
                    from sklearn.metrics import accuracy_score
                    score = accuracy_score(y_val, pred)
                else:
                    from sklearn.metrics import mean_squared_error
                    score = -mean_squared_error(y_val, pred)
                scores.append(score)
        else:
            scores = []
            for model in self.models:
                if self.config.target_type == 'classification':
                    score = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()
                else:
                    score = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error').mean()
                scores.append(score)
        
        scores = np.array(scores)
        scores = scores - scores.min() + 1e-8
        self.weights = scores / scores.sum()
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.ensemble_method == 'stacking':
            return self._predict_stacking(X)
        elif self.ensemble_method == 'weighted':
            return self._predict_weighted(X)
        else:
            return self._predict_average(X)
    
    def _predict_stacking(self, X: pd.DataFrame) -> np.ndarray:
        meta_features = []
        
        for model in self.models:
            if self.config.target_type == 'classification':
                pred = model.predict_proba(X)
                if len(pred.shape) > 1:
                    pred = pred[:, 1]
            else:
                pred = model.predict(X)
            meta_features.append(pred)
        
        meta_X = np.column_stack(meta_features)
        return self.meta_model.predict(meta_X)
    
    def _predict_weighted(self, X: pd.DataFrame) -> np.ndarray:
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        return np.average(predictions, axis=0, weights=self.weights)
    
    def _predict_average(self, X: pd.DataFrame) -> np.ndarray:
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        return np.mean(predictions, axis=0)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.config.target_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.ensemble_method == 'stacking':
            meta_features = []
            
            for model in self.models:
                pred = model.predict_proba(X)
                if len(pred.shape) > 1:
                    pred = pred[:, 1]
                meta_features.append(pred)
            
            meta_X = np.column_stack(meta_features)
            return self.meta_model.predict_proba(meta_X)
        else:
            probas = []
            
            for model in self.models:
                proba = model.predict_proba(X)
                probas.append(proba)
            
            if self.ensemble_method == 'weighted':
                return np.average(probas, axis=0, weights=self.weights)
            else:
                return np.mean(probas, axis=0)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self.fitted or not self.feature_names:
            return None
        
        importances = {}
        
        for model in self.models:
            model_importance = model.get_feature_importance()
            if model_importance:
                for feature, importance in model_importance.items():
                    if feature not in importances:
                        importances[feature] = []
                    importances[feature].append(importance)
        
        avg_importances = {}
        for feature, importance_list in importances.items():
            avg_importances[feature] = np.mean(importance_list)
        
        return avg_importances


class VotingEnsemble(BaseModel):
    def __init__(self, models: List[BaseModel], voting: str = 'soft'):
        self.models = models
        self.voting = voting
        self.sklearn_ensemble = None
        
        config = ModelConfig(
            name=f'voting_{voting}',
            model_type='voting',
            params={'voting': voting},
            target_type=models[0].config.target_type if models else 'regression'
        )
        super().__init__(config)
    
    def _create_model(self):
        estimators = [(model.config.name, model.model) for model in self.models]
        
        if self.config.target_type == 'classification':
            return VotingClassifier(estimators=estimators, voting=self.voting)
        else:
            return VotingRegressor(estimators=estimators)
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None):
        
        self.feature_names = X.columns.tolist()
        
        for model in self.models:
            model.fit(X, y, X_val, y_val)
        
        self.model = self._create_model()
        self.model.fit(X, y)
        
        self.fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.config.target_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)


class StackingEnsemble(BaseModel):
    def __init__(self, base_models: List[BaseModel], meta_model: Optional[BaseModel] = None,
                 cv_folds: int = 5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv_folds = cv_folds
        self.fitted_base_models = []
        self.fitted_meta_model = None
        
        config = ModelConfig(
            name='stacking',
            model_type='stacking',
            params={'cv_folds': cv_folds},
            target_type=base_models[0].config.target_type if base_models else 'regression'
        )
        super().__init__(config)
    
    def _create_model(self):
        return None
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None):
        
        self.feature_names = X.columns.tolist()
        
        from sklearn.model_selection import KFold, StratifiedKFold
        
        if self.config.target_type == 'classification':
            kfold = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        else:
            kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        meta_features = np.zeros((len(X), len(self.base_models)))
        
        for i, model in enumerate(self.base_models):
            print(f"Training base model {i+1}/{len(self.base_models)}: {model.config.name}")
            
            fold_predictions = np.zeros(len(X))
            
            for train_idx, val_idx in kfold.split(X, y):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                
                model_copy = type(model)(model.config)
                model_copy.fit(X_train_fold, y_train_fold)
                
                if self.config.target_type == 'classification':
                    pred = model_copy.predict_proba(X_val_fold)
                    if len(pred.shape) > 1:
                        pred = pred[:, 1]
                else:
                    pred = model_copy.predict(X_val_fold)
                
                fold_predictions[val_idx] = pred
            
            meta_features[:, i] = fold_predictions
            
            model.fit(X, y, X_val, y_val)
            self.fitted_base_models.append(model)
        
        if self.meta_model is None:
            if self.config.target_type == 'classification':
                from .linear_models import LogisticModel, create_ridge_config
                config = create_ridge_config(target_type='classification')
                self.meta_model = LogisticModel(config)
            else:
                from .linear_models import RidgeModel, create_ridge_config
                config = create_ridge_config(target_type='regression')
                self.meta_model = RidgeModel(config)
        
        print("Training meta model...")
        meta_X = pd.DataFrame(meta_features, columns=[f'base_{i}' for i in range(len(self.base_models))])
        self.fitted_meta_model = self.meta_model
        self.fitted_meta_model.fit(meta_X, y)
        
        self.fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        meta_features = np.zeros((len(X), len(self.fitted_base_models)))
        
        for i, model in enumerate(self.fitted_base_models):
            if self.config.target_type == 'classification':
                pred = model.predict_proba(X)
                if len(pred.shape) > 1:
                    pred = pred[:, 1]
            else:
                pred = model.predict(X)
            meta_features[:, i] = pred
        
        meta_X = pd.DataFrame(meta_features, columns=[f'base_{i}' for i in range(len(self.fitted_base_models))])
        return self.fitted_meta_model.predict(meta_X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.config.target_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        meta_features = np.zeros((len(X), len(self.fitted_base_models)))
        
        for i, model in enumerate(self.fitted_base_models):
            pred = model.predict_proba(X)
            if len(pred.shape) > 1:
                pred = pred[:, 1]
            meta_features[:, i] = pred
        
        meta_X = pd.DataFrame(meta_features, columns=[f'base_{i}' for i in range(len(self.fitted_base_models))])
        return self.fitted_meta_model.predict_proba(meta_X)


def create_ensemble_from_models(models: List[BaseModel], 
                               ensemble_type: str = 'average') -> BaseModel:
    
    if ensemble_type == 'average':
        return EnsembleModel(models, ensemble_method='average')
    elif ensemble_type == 'weighted':
        return EnsembleModel(models, ensemble_method='weighted')
    elif ensemble_type == 'stacking':
        return StackingEnsemble(models)
    elif ensemble_type == 'voting_soft':
        return VotingEnsemble(models, voting='soft')
    elif ensemble_type == 'voting_hard':
        return VotingEnsemble(models, voting='hard')
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")


def create_diverse_ensemble(target_type: str = 'regression') -> List[BaseModel]:
    from .factory import create_kaggle_models
    
    models = create_kaggle_models(target_type)
    
    ensemble_models = []
    ensemble_models.append(create_ensemble_from_models(models[:3], 'average'))
    ensemble_models.append(create_ensemble_from_models(models[3:6], 'weighted'))
    ensemble_models.append(create_ensemble_from_models(models, 'stacking'))
    
    return ensemble_models