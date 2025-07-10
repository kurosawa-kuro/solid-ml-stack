import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from itertools import product
from sklearn.model_selection import ParameterGrid
from .base import BaseOptimizer, OptimizationConfig, OptimizationResult
import time


class GridSearchOptimizer(BaseOptimizer):
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.param_grid = self._create_param_grid()
        
    def _create_param_grid(self) -> List[Dict[str, Any]]:
        return list(ParameterGrid(self.config.search_space))
    
    def optimize(self, 
                model,
                X: pd.DataFrame,
                y: pd.Series,
                X_val: Optional[pd.DataFrame] = None,
                y_val: Optional[pd.Series] = None) -> OptimizationResult:
        
        print(f"Starting Grid Search with {len(self.param_grid)} parameter combinations")
        
        start_time = time.time()
        
        for trial_number, params in enumerate(self.param_grid):
            if self.config.timeout and (time.time() - start_time) > self.config.timeout:
                print(f"Timeout reached after {trial_number} trials")
                break
                
            if self.config.n_trials and trial_number >= self.config.n_trials:
                print(f"Maximum trials ({self.config.n_trials}) reached")
                break
            
            try:
                score = self._evaluate_model(model, params, X, y, X_val, y_val)
                self._log_trial(trial_number, params, score)
                
                if trial_number % 10 == 0:
                    print(f"Trial {trial_number}: Score = {score:.4f}")
                    
            except Exception as e:
                print(f"Trial {trial_number} failed: {e}")
                continue
        
        end_time = time.time()
        print(f"Grid Search completed in {end_time - start_time:.2f} seconds")
        print(f"Best score: {self.best_score_:.4f}")
        print(f"Best parameters: {self.best_params_}")
        
        return OptimizationResult(self, model)
    
    def get_param_importance(self) -> Dict[str, float]:
        if not self.optimization_history_:
            return {}
        
        param_effects = {}
        
        for param_name in self.config.search_space.keys():
            scores_by_param = {}
            
            for trial in self.optimization_history_:
                param_value = trial['params'].get(param_name)
                if param_value not in scores_by_param:
                    scores_by_param[param_value] = []
                scores_by_param[param_value].append(trial['score'])
            
            if len(scores_by_param) > 1:
                param_means = {k: np.mean(v) for k, v in scores_by_param.items()}
                param_effects[param_name] = np.std(list(param_means.values()))
            else:
                param_effects[param_name] = 0.0
        
        total_effect = sum(param_effects.values())
        if total_effect > 0:
            param_effects = {k: v/total_effect for k, v in param_effects.items()}
        
        return param_effects


def create_xgboost_grid_search(target_type: str = 'regression') -> OptimizationConfig:
    if target_type == 'classification':
        search_space = {
            'n_estimators': [100, 500, 1000],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0, 0.1, 1]
        }
        scoring = 'f1_weighted'
    else:
        search_space = {
            'n_estimators': [100, 500, 1000],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0, 0.1, 1]
        }
        scoring = 'neg_mean_squared_error'
    
    return OptimizationConfig(
        search_space=search_space,
        n_trials=200,
        cv_folds=5,
        scoring=scoring,
        direction='maximize' if target_type == 'classification' else 'maximize'
    )


def create_lightgbm_grid_search(target_type: str = 'regression') -> OptimizationConfig:
    if target_type == 'classification':
        search_space = {
            'n_estimators': [100, 500, 1000],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0, 0.1, 1],
            'num_leaves': [31, 62, 127]
        }
        scoring = 'f1_weighted'
    else:
        search_space = {
            'n_estimators': [100, 500, 1000],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0, 0.1, 1],
            'num_leaves': [31, 62, 127]
        }
        scoring = 'neg_mean_squared_error'
    
    return OptimizationConfig(
        search_space=search_space,
        n_trials=200,
        cv_folds=5,
        scoring=scoring,
        direction='maximize' if target_type == 'classification' else 'maximize'
    )


def create_catboost_grid_search(target_type: str = 'regression') -> OptimizationConfig:
    if target_type == 'classification':
        search_space = {
            'iterations': [100, 500, 1000],
            'depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.1, 0.2],
            'l2_leaf_reg': [1, 3, 5, 7, 9],
            'border_count': [32, 64, 128]
        }
        scoring = 'f1_weighted'
    else:
        search_space = {
            'iterations': [100, 500, 1000],
            'depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.1, 0.2],
            'l2_leaf_reg': [1, 3, 5, 7, 9],
            'border_count': [32, 64, 128]
        }
        scoring = 'neg_mean_squared_error'
    
    return OptimizationConfig(
        search_space=search_space,
        n_trials=200,
        cv_folds=5,
        scoring=scoring,
        direction='maximize' if target_type == 'classification' else 'maximize'
    )