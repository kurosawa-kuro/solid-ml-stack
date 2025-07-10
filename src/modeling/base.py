import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import joblib
import json
from pathlib import Path


@dataclass
class ModelConfig:
    name: str
    model_type: str
    params: Dict[str, Any]
    target_type: str = 'regression'
    random_state: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'model_type': self.model_type,
            'params': self.params,
            'target_type': self.target_type,
            'random_state': self.random_state
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        return cls(**config_dict)
        
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load(cls, filepath: str) -> 'ModelConfig':
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class BaseModel(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.fitted = False
        self.feature_names = None
        self.feature_importances_ = None
        
    @abstractmethod
    def _create_model(self):
        pass
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, 
            y_val: Optional[pd.Series] = None):
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.config.target_type == 'classification':
            return self.model.predict_proba(X)
        else:
            raise ValueError("predict_proba only available for classification")
            
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if self.feature_importances_ is not None and self.feature_names is not None:
            return dict(zip(self.feature_names, self.feature_importances_))
        return None
        
    def save(self, filepath: str):
        if not self.fitted:
            raise ValueError("Model must be fitted before saving")
            
        save_dict = {
            'model': self.model,
            'config': self.config.to_dict(),
            'feature_names': self.feature_names,
            'feature_importances': self.feature_importances_,
            'fitted': self.fitted
        }
        
        joblib.dump(save_dict, filepath)
        
    def load(self, filepath: str):
        save_dict = joblib.load(filepath)
        
        self.model = save_dict['model']
        self.config = ModelConfig.from_dict(save_dict['config'])
        self.feature_names = save_dict['feature_names']
        self.feature_importances_ = save_dict['feature_importances']
        self.fitted = save_dict['fitted']
        
    def get_params(self) -> Dict[str, Any]:
        return self.config.params.copy()
        
    def set_params(self, **params):
        self.config.params.update(params)
        if self.model is not None:
            self.model.set_params(**params)
        return self
        
    def __str__(self) -> str:
        return f"{self.config.name} ({self.config.model_type})"
        
    def __repr__(self) -> str:
        return self.__str__()


class ModelPipeline:
    def __init__(self, models: List[BaseModel]):
        self.models = models
        self.fitted_models = []
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None):
        
        self.feature_names = X.columns.tolist()
        self.fitted_models = []
        
        for model in self.models:
            print(f"Training {model.config.name}...")
            model.fit(X, y, X_val, y_val)
            self.fitted_models.append(model)
            
        return self
        
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        predictions = {}
        for model in self.fitted_models:
            predictions[model.config.name] = model.predict(X)
        return predictions
        
    def predict_proba(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        predictions = {}
        for model in self.fitted_models:
            try:
                predictions[model.config.name] = model.predict_proba(X)
            except:
                pass
        return predictions
        
    def get_feature_importances(self) -> Dict[str, Dict[str, float]]:
        importances = {}
        for model in self.fitted_models:
            importance = model.get_feature_importance()
            if importance is not None:
                importances[model.config.name] = importance
        return importances
        
    def save(self, directory: str):
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        for i, model in enumerate(self.fitted_models):
            filepath = Path(directory) / f"{model.config.name}_{i}.pkl"
            model.save(str(filepath))
            
        metadata = {
            'n_models': len(self.fitted_models),
            'model_names': [model.config.name for model in self.fitted_models],
            'feature_names': self.feature_names
        }
        
        with open(Path(directory) / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def load(self, directory: str):
        with open(Path(directory) / 'metadata.json', 'r') as f:
            metadata = json.load(f)
            
        self.fitted_models = []
        self.feature_names = metadata['feature_names']
        
        for i, model_name in enumerate(metadata['model_names']):
            filepath = Path(directory) / f"{model_name}_{i}.pkl"
            model = BaseModel.__new__(BaseModel)
            model.load(str(filepath))
            self.fitted_models.append(model)