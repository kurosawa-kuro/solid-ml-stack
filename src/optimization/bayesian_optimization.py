import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from .base import BaseOptimizer, OptimizationConfig, OptimizationResult
import time
import warnings
warnings.filterwarnings('ignore')


class BayesianOptimizer(BaseOptimizer):
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            from skopt.utils import use_named_args
            self.gp_minimize = gp_minimize
            self.Real = Real
            self.Integer = Integer
            self.Categorical = Categorical
            self.use_named_args = use_named_args
            self.available = True
        except ImportError:
            print("scikit-optimize not available. Using random search instead.")
            self.available = False
        
        self.search_space_skopt = self._create_search_space()
        self.dimension_names = []
        
    def _create_search_space(self):
        if not self.available:
            return None
            
        dimensions = []
        self.dimension_names = []
        
        for param_name, param_config in self.config.search_space.items():
            self.dimension_names.append(param_name)
            
            if isinstance(param_config, list):
                if all(isinstance(x, (int, float)) for x in param_config):
                    dimensions.append(self.Real(min(param_config), max(param_config), name=param_name))
                else:
                    dimensions.append(self.Categorical(param_config, name=param_name))
            elif isinstance(param_config, dict):
                if param_config.get('type') == 'uniform':
                    dimensions.append(self.Real(param_config['low'], param_config['high'], name=param_name))
                elif param_config.get('type') == 'loguniform':
                    dimensions.append(self.Real(param_config['low'], param_config['high'], prior='log-uniform', name=param_name))
                elif param_config.get('type') == 'int':
                    dimensions.append(self.Integer(param_config['low'], param_config['high'], name=param_name))
                elif param_config.get('type') == 'choice':
                    dimensions.append(self.Categorical(param_config['choices'], name=param_name))
                else:
                    dimensions.append(self.Categorical([param_config], name=param_name))
            else:
                dimensions.append(self.Categorical([param_config], name=param_name))
                
        return dimensions
    
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
        
        print(f"Starting Bayesian Optimization with {self.config.n_trials} trials")
        
        @self.use_named_args(self.search_space_skopt)
        def objective(**params):
            try:
                score = self._evaluate_model(model, params, X, y, X_val, y_val)
                
                trial_number = len(self.optimization_history_)
                self._log_trial(trial_number, params, score)
                
                if trial_number % 10 == 0:
                    print(f"Trial {trial_number}: Score = {score:.4f}, Best = {self.best_score_:.4f}")
                
                return -score if self.config.direction == 'maximize' else score
                
            except Exception as e:
                print(f"Trial failed: {e}")
                return float('inf')
        
        start_time = time.time()
        
        try:
            result = self.gp_minimize(
                func=objective,
                dimensions=self.search_space_skopt,
                n_calls=self.config.n_trials,
                n_initial_points=min(10, self.config.n_trials // 4),
                random_state=self.config.random_state,
                verbose=False
            )
            
            self.study_results_ = result
            
        except Exception as e:
            print(f"Bayesian optimization failed: {e}")
            print("Falling back to random search")
            from .random_search import RandomSearchOptimizer
            fallback_optimizer = RandomSearchOptimizer(self.config)
            return fallback_optimizer.optimize(model, X, y, X_val, y_val)
        
        end_time = time.time()
        print(f"Bayesian Optimization completed in {end_time - start_time:.2f} seconds")
        print(f"Best score: {self.best_score_:.4f}")
        print(f"Best parameters: {self.best_params_}")
        
        return OptimizationResult(self, model)
    
    def get_param_importance(self) -> Optional[Dict[str, float]]:
        if not self.available or self.study_results_ is None:
            return None
        
        try:
            from skopt.plots import plot_objective
            
            importance_dict = {}
            
            for i, param_name in enumerate(self.dimension_names):
                if hasattr(self.study_results_, 'func_vals'):
                    param_values = [x[i] for x in self.study_results_.x_iters]
                    scores = self.study_results_.func_vals
                    
                    if len(set(param_values)) > 1:
                        correlation = np.corrcoef(
                            [float(v) if isinstance(v, (int, float)) else hash(str(v)) for v in param_values],
                            scores
                        )[0, 1]
                        importance_dict[param_name] = abs(correlation) if not np.isnan(correlation) else 0.0
                    else:
                        importance_dict[param_name] = 0.0
            
            total_importance = sum(importance_dict.values())
            if total_importance > 0:
                importance_dict = {k: v/total_importance for k, v in importance_dict.items()}
            
            return importance_dict
            
        except Exception as e:
            print(f"Could not compute parameter importance: {e}")
            return None
    
    def plot_convergence(self):
        if not self.available or self.study_results_ is None:
            print("No results available for plotting")
            return
        
        try:
            from skopt.plots import plot_convergence
            import matplotlib.pyplot as plt
            
            plot_convergence(self.study_results_)
            plt.show()
            
        except ImportError:
            print("matplotlib not available for plotting")
        except Exception as e:
            print(f"Could not plot convergence: {e}")
    
    def plot_evaluations(self):
        if not self.available or self.study_results_ is None:
            print("No results available for plotting")
            return
        
        try:
            from skopt.plots import plot_evaluations
            import matplotlib.pyplot as plt
            
            plot_evaluations(self.study_results_, dimension_names=self.dimension_names)
            plt.show()
            
        except ImportError:
            print("matplotlib not available for plotting")
        except Exception as e:
            print(f"Could not plot evaluations: {e}")


def create_xgboost_bayesian_search(target_type: str = 'regression') -> OptimizationConfig:
    if target_type == 'classification':
        search_space = {
            'n_estimators': {'type': 'int', 'low': 100, 'high': 2000},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10},
            'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
            'subsample': {'type': 'uniform', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'uniform', 'low': 0.6, 'high': 1.0},
            'reg_alpha': {'type': 'loguniform', 'low': 0.01, 'high': 10},
            'reg_lambda': {'type': 'loguniform', 'low': 0.01, 'high': 10}
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
            'reg_lambda': {'type': 'loguniform', 'low': 0.01, 'high': 10}
        }
        scoring = 'neg_mean_squared_error'
    
    return OptimizationConfig(
        search_space=search_space,
        n_trials=100,
        cv_folds=5,
        scoring=scoring,
        direction='maximize' if target_type == 'classification' else 'maximize'
    )