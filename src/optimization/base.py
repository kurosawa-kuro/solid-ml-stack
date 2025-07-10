import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
import time


@dataclass
class OptimizationConfig:
    search_space: Dict[str, Any]
    n_trials: int = 100
    cv_folds: int = 5
    scoring: str = 'neg_mean_squared_error'
    random_state: int = 42
    n_jobs: int = -1
    timeout: Optional[int] = None
    early_stopping_rounds: Optional[int] = None
    direction: str = 'minimize'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'search_space': self.search_space,
            'n_trials': self.n_trials,
            'cv_folds': self.cv_folds,
            'scoring': self.scoring,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'timeout': self.timeout,
            'early_stopping_rounds': self.early_stopping_rounds,
            'direction': self.direction
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OptimizationConfig':
        return cls(**config_dict)


class BaseOptimizer(ABC):
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.best_params_ = None
        self.best_score_ = None
        self.optimization_history_ = []
        self.study_results_ = None
        
    @abstractmethod
    def optimize(self, 
                model,
                X: pd.DataFrame,
                y: pd.Series,
                X_val: Optional[pd.DataFrame] = None,
                y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        pass
    
    def _evaluate_model(self, model, params: Dict[str, Any],
                       X: pd.DataFrame, y: pd.Series,
                       X_val: Optional[pd.DataFrame] = None,
                       y_val: Optional[pd.Series] = None) -> float:
        
        model.set_params(**params)
        
        if X_val is not None and y_val is not None:
            model.fit(X, y, X_val, y_val)
            y_pred = model.predict(X_val)
            
            if self.config.scoring == 'neg_mean_squared_error':
                return -mean_squared_error(y_val, y_pred)
            elif self.config.scoring == 'neg_mean_absolute_error':
                return -mean_absolute_error(y_val, y_pred)
            elif self.config.scoring == 'r2':
                return r2_score(y_val, y_pred)
            elif self.config.scoring == 'accuracy':
                return accuracy_score(y_val, y_pred)
            elif self.config.scoring == 'f1':
                return f1_score(y_val, y_pred, average='weighted')
            elif self.config.scoring == 'roc_auc':
                y_proba = model.predict_proba(X_val)
                if len(y_proba.shape) > 1:
                    y_proba = y_proba[:, 1]
                return roc_auc_score(y_val, y_proba)
        else:
            scores = cross_val_score(
                model, X, y,
                cv=self.config.cv_folds,
                scoring=self.config.scoring,
                n_jobs=self.config.n_jobs
            )
            return scores.mean()
    
    def _log_trial(self, trial_number: int, params: Dict[str, Any], score: float):
        self.optimization_history_.append({
            'trial': trial_number,
            'params': params.copy(),
            'score': score,
            'timestamp': time.time()
        })
        
        if self.best_score_ is None or (
            (self.config.direction == 'maximize' and score > self.best_score_) or
            (self.config.direction == 'minimize' and score < self.best_score_)
        ):
            self.best_score_ = score
            self.best_params_ = params.copy()
    
    def get_best_params(self) -> Dict[str, Any]:
        if self.best_params_ is None:
            raise ValueError("No optimization has been performed yet")
        return self.best_params_.copy()
    
    def get_best_score(self) -> float:
        if self.best_score_ is None:
            raise ValueError("No optimization has been performed yet")
        return self.best_score_
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        return self.optimization_history_.copy()
    
    def save_results(self, filepath: str):
        results = {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'optimization_history': self.optimization_history_,
            'config': self.config.to_dict()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def load_results(self, filepath: str):
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        self.best_params_ = results['best_params']
        self.best_score_ = results['best_score']
        self.optimization_history_ = results['optimization_history']
        self.config = OptimizationConfig.from_dict(results['config'])
    
    def plot_optimization_history(self):
        try:
            import matplotlib.pyplot as plt
            
            if not self.optimization_history_:
                print("No optimization history to plot")
                return
            
            trials = [h['trial'] for h in self.optimization_history_]
            scores = [h['score'] for h in self.optimization_history_]
            
            plt.figure(figsize=(10, 6))
            plt.plot(trials, scores, 'b-', alpha=0.7)
            plt.xlabel('Trial')
            plt.ylabel('Score')
            plt.title('Optimization History')
            plt.grid(True, alpha=0.3)
            plt.show()
            
        except ImportError:
            print("matplotlib not available for plotting")
    
    def get_param_importance(self) -> Optional[Dict[str, float]]:
        return None


class OptimizationResult:
    def __init__(self, optimizer: BaseOptimizer, model):
        self.optimizer = optimizer
        self.model = model
        self.best_params = optimizer.get_best_params()
        self.best_score = optimizer.get_best_score()
        self.optimization_history = optimizer.get_optimization_history()
        
    def apply_best_params(self):
        self.model.set_params(**self.best_params)
        return self.model
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            'best_score': self.best_score,
            'best_params': self.best_params,
            'n_trials': len(self.optimization_history),
            'optimization_time': self._get_optimization_time()
        }
    
    def _get_optimization_time(self) -> float:
        if not self.optimization_history:
            return 0.0
        
        start_time = self.optimization_history[0]['timestamp']
        end_time = self.optimization_history[-1]['timestamp']
        return end_time - start_time