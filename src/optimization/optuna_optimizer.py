import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from .base import BaseOptimizer, OptimizationConfig, OptimizationResult
import time
import warnings
warnings.filterwarnings('ignore')


class OptunaOptimizer(BaseOptimizer):
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        try:
            import optuna
            self.optuna = optuna
            self.available = True
        except ImportError:
            print("Optuna not available. Using random search instead.")
            self.available = False
        
        self.study = None
        
    def _create_objective(self, model, X, y, X_val, y_val):
        def objective(trial):
            try:
                params = self._suggest_params(trial)
                score = self._evaluate_model(model, params, X, y, X_val, y_val)
                
                trial_number = len(self.optimization_history_)
                self._log_trial(trial_number, params, score)
                
                if trial_number % 10 == 0:
                    print(f"Trial {trial_number}: Score = {score:.4f}, Best = {self.best_score_:.4f}")
                
                return -score if self.config.direction == 'maximize' else score
                
            except Exception as e:
                print(f"Trial failed: {e}")
                return float('inf')
        
        return objective
    
    def _suggest_params(self, trial):
        params = {}
        
        for param_name, param_config in self.config.search_space.items():
            if isinstance(param_config, list):
                if all(isinstance(x, (int, float)) for x in param_config):
                    if all(isinstance(x, int) for x in param_config):
                        params[param_name] = trial.suggest_int(param_name, min(param_config), max(param_config))
                    else:
                        params[param_name] = trial.suggest_float(param_name, min(param_config), max(param_config))
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_config)
            elif isinstance(param_config, dict):
                if param_config.get('type') == 'uniform':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
                elif param_config.get('type') == 'loguniform':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)
                elif param_config.get('type') == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
                elif param_config.get('type') == 'choice':
                    params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
                else:
                    params[param_name] = trial.suggest_categorical(param_name, [param_config])
            else:
                params[param_name] = param_config
                
        return params
    
    def optimize(self, 
                model,
                X: pd.DataFrame,
                y: pd.Series,
                X_val: Optional[pd.DataFrame] = None,
                y_val: Optional[pd.Series] = None) -> OptimizationResult:
        
        if not self.available:
            from .random_search import RandomSearchOptimizer
            print("Using RandomSearchOptimizer as fallback")
            fallback_optimizer = RandomSearchOptimizer(self.config)
            return fallback_optimizer.optimize(model, X, y, X_val, y_val)
        
        print(f"Starting Optuna Optimization with {self.config.n_trials} trials")
        
        direction = 'minimize' if self.config.direction == 'minimize' else 'maximize'
        
        self.study = self.optuna.create_study(
            direction=direction,
            sampler=self.optuna.samplers.TPESampler(seed=self.config.random_state)
        )
        
        objective = self._create_objective(model, X, y, X_val, y_val)
        
        start_time = time.time()
        
        try:
            self.study.optimize(
                objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout,
                show_progress_bar=False
            )
            
        except Exception as e:
            print(f"Optuna optimization failed: {e}")
            print("Falling back to random search")
            from .random_search import RandomSearchOptimizer
            fallback_optimizer = RandomSearchOptimizer(self.config)
            return fallback_optimizer.optimize(model, X, y, X_val, y_val)
        
        end_time = time.time()
        print(f"Optuna Optimization completed in {end_time - start_time:.2f} seconds")
        print(f"Best score: {self.best_score_:.4f}")
        print(f"Best parameters: {self.best_params_}")
        
        return OptimizationResult(self, model)
    
    def get_param_importance(self) -> Optional[Dict[str, float]]:
        if not self.available or self.study is None:
            return None
        
        try:
            importance = self.optuna.importance.get_param_importances(self.study)
            return dict(importance)
            
        except Exception as e:
            print(f"Could not compute parameter importance: {e}")
            return None
    
    def plot_optimization_history(self):
        if not self.available or self.study is None:
            print("No study available for plotting")
            return
        
        try:
            from optuna.visualization import plot_optimization_history
            import matplotlib.pyplot as plt
            
            fig = plot_optimization_history(self.study)
            fig.show()
            
        except ImportError:
            print("Visualization libraries not available")
        except Exception as e:
            print(f"Could not plot optimization history: {e}")
    
    def plot_param_importances(self):
        if not self.available or self.study is None:
            print("No study available for plotting")
            return
        
        try:
            from optuna.visualization import plot_param_importances
            import matplotlib.pyplot as plt
            
            fig = plot_param_importances(self.study)
            fig.show()
            
        except ImportError:
            print("Visualization libraries not available")
        except Exception as e:
            print(f"Could not plot parameter importances: {e}")
    
    def plot_parallel_coordinate(self):
        if not self.available or self.study is None:
            print("No study available for plotting")
            return
        
        try:
            from optuna.visualization import plot_parallel_coordinate
            import matplotlib.pyplot as plt
            
            fig = plot_parallel_coordinate(self.study)
            fig.show()
            
        except ImportError:
            print("Visualization libraries not available")
        except Exception as e:
            print(f"Could not plot parallel coordinate: {e}")


def create_xgboost_optuna_search(target_type: str = 'regression') -> OptimizationConfig:
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
        n_trials=100,
        cv_folds=5,
        scoring=scoring,
        direction='maximize' if target_type == 'classification' else 'maximize'
    )