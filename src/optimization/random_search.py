import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from sklearn.model_selection import RandomizedSearchCV
from .base import BaseOptimizer, OptimizationConfig, OptimizationResult
import time
import random


class RandomSearchOptimizer(BaseOptimizer):
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        
    def _sample_params(self) -> Dict[str, Any]:
        params = {}
        
        for param_name, param_values in self.config.search_space.items():
            if isinstance(param_values, list):
                params[param_name] = random.choice(param_values)
            elif isinstance(param_values, dict):
                if param_values.get('type') == 'uniform':
                    params[param_name] = random.uniform(
                        param_values['low'], param_values['high']
                    )
                elif param_values.get('type') == 'loguniform':
                    params[param_name] = np.exp(random.uniform(
                        np.log(param_values['low']), np.log(param_values['high'])
                    ))
                elif param_values.get('type') == 'int':
                    params[param_name] = random.randint(
                        param_values['low'], param_values['high']
                    )
                elif param_values.get('type') == 'choice':
                    params[param_name] = random.choice(param_values['choices'])
                else:
                    params[param_name] = random.choice(param_values)
            else:
                params[param_name] = param_values
                
        return params
    
    def optimize(self, 
                model,
                X: pd.DataFrame,
                y: pd.Series,
                X_val: Optional[pd.DataFrame] = None,
                y_val: Optional[pd.Series] = None) -> OptimizationResult:
        
        print(f"Starting Random Search with {self.config.n_trials} trials")
        
        start_time = time.time()
        no_improvement_count = 0
        
        for trial_number in range(self.config.n_trials):
            if self.config.timeout and (time.time() - start_time) > self.config.timeout:
                print(f"Timeout reached after {trial_number} trials")
                break
            
            try:
                params = self._sample_params()
                score = self._evaluate_model(model, params, X, y, X_val, y_val)
                
                old_best_score = self.best_score_
                self._log_trial(trial_number, params, score)
                
                if self.best_score_ == old_best_score:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
                
                if (self.config.early_stopping_rounds and 
                    no_improvement_count >= self.config.early_stopping_rounds):
                    print(f"Early stopping after {trial_number} trials")
                    break
                
                if trial_number % 10 == 0:
                    print(f"Trial {trial_number}: Score = {score:.4f}, Best = {self.best_score_:.4f}")
                    
            except Exception as e:
                print(f"Trial {trial_number} failed: {e}")
                continue
        
        end_time = time.time()
        print(f"Random Search completed in {end_time - start_time:.2f} seconds")
        print(f"Best score: {self.best_score_:.4f}")
        print(f"Best parameters: {self.best_params_}")
        
        return OptimizationResult(self, model)
    
    def get_param_importance(self) -> Dict[str, float]:
        if not self.optimization_history_:
            return {}
        
        param_effects = {}
        
        for param_name in self.config.search_space.keys():
            param_values = [trial['params'].get(param_name) for trial in self.optimization_history_]
            scores = [trial['score'] for trial in self.optimization_history_]
            
            if len(set(param_values)) > 1:
                correlation = np.corrcoef(
                    [float(v) if isinstance(v, (int, float)) else hash(str(v)) for v in param_values],
                    scores
                )[0, 1]
                param_effects[param_name] = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                param_effects[param_name] = 0.0
        
        total_effect = sum(param_effects.values())
        if total_effect > 0:
            param_effects = {k: v/total_effect for k, v in param_effects.items()}
        
        return param_effects


def create_xgboost_random_search(target_type: str = 'regression') -> OptimizationConfig:
    if target_type == 'classification':
        search_space = {
            'n_estimators': {'type': 'int', 'low': 100, 'high': 2000},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10},
            'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
            'subsample': {'type': 'uniform', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'uniform', 'low': 0.6, 'high': 1.0},
            'reg_alpha': {'type': 'loguniform', 'low': 0.01, 'high': 10},
            'reg_lambda': {'type': 'loguniform', 'low': 0.01, 'high': 10},
            'gamma': {'type': 'uniform', 'low': 0, 'high': 5}
        }
        scoring = 'f1_weighted'
    else:
        search_space = {
            'n_estimators': {'type': 'int', 'low': 100, 'high': 2000},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10},
            'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
            'subsample': {'type': 'uniform', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'uniform', 'low': 0.6, 'high': 1.0},
            'reg_alpha': {'type': 'loguniform', 'low': 0.01, 'high': 10},
            'reg_lambda': {'type': 'loguniform', 'low': 0.01, 'high': 10},
            'gamma': {'type': 'uniform', 'low': 0, 'high': 5}
        }
        scoring = 'neg_mean_squared_error'
    
    return OptimizationConfig(
        search_space=search_space,
        n_trials=200,
        cv_folds=5,
        scoring=scoring,
        direction='maximize' if target_type == 'classification' else 'maximize',
        early_stopping_rounds=50
    )


def create_lightgbm_random_search(target_type: str = 'regression') -> OptimizationConfig:
    if target_type == 'classification':
        search_space = {
            'n_estimators': {'type': 'int', 'low': 100, 'high': 2000},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10},
            'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
            'subsample': {'type': 'uniform', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'uniform', 'low': 0.6, 'high': 1.0},
            'reg_alpha': {'type': 'loguniform', 'low': 0.01, 'high': 10},
            'reg_lambda': {'type': 'loguniform', 'low': 0.01, 'high': 10},
            'num_leaves': {'type': 'int', 'low': 10, 'high': 300},
            'min_child_samples': {'type': 'int', 'low': 5, 'high': 100}
        }
        scoring = 'f1_weighted'
    else:
        search_space = {
            'n_estimators': {'type': 'int', 'low': 100, 'high': 2000},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10},
            'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
            'subsample': {'type': 'uniform', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'uniform', 'low': 0.6, 'high': 1.0},
            'reg_alpha': {'type': 'loguniform', 'low': 0.01, 'high': 10},
            'reg_lambda': {'type': 'loguniform', 'low': 0.01, 'high': 10},
            'num_leaves': {'type': 'int', 'low': 10, 'high': 300},
            'min_child_samples': {'type': 'int', 'low': 5, 'high': 100}
        }
        scoring = 'neg_mean_squared_error'
    
    return OptimizationConfig(
        search_space=search_space,
        n_trials=200,
        cv_folds=5,
        scoring=scoring,
        direction='maximize' if target_type == 'classification' else 'maximize',
        early_stopping_rounds=50
    )


def create_catboost_random_search(target_type: str = 'regression') -> OptimizationConfig:
    if target_type == 'classification':
        search_space = {
            'iterations': {'type': 'int', 'low': 100, 'high': 2000},
            'depth': {'type': 'int', 'low': 3, 'high': 10},
            'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
            'l2_leaf_reg': {'type': 'loguniform', 'low': 0.1, 'high': 10},
            'border_count': {'type': 'int', 'low': 32, 'high': 255},
            'bagging_temperature': {'type': 'uniform', 'low': 0, 'high': 1}
        }
        scoring = 'f1_weighted'
    else:
        search_space = {
            'iterations': {'type': 'int', 'low': 100, 'high': 2000},
            'depth': {'type': 'int', 'low': 3, 'high': 10},
            'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
            'l2_leaf_reg': {'type': 'loguniform', 'low': 0.1, 'high': 10},
            'border_count': {'type': 'int', 'low': 32, 'high': 255},
            'bagging_temperature': {'type': 'uniform', 'low': 0, 'high': 1}
        }
        scoring = 'neg_mean_squared_error'
    
    return OptimizationConfig(
        search_space=search_space,
        n_trials=200,
        cv_folds=5,
        scoring=scoring,
        direction='maximize' if target_type == 'classification' else 'maximize',
        early_stopping_rounds=50
    )