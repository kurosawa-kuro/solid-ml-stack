from typing import Dict, Any, Optional, List
from .base import BaseOptimizer, OptimizationConfig
from .grid_search import GridSearchOptimizer
from .random_search import RandomSearchOptimizer
from .bayesian_optimization import BayesianOptimizer
from .optuna_optimizer import OptunaOptimizer


class OptimizerFactory:
    _optimizer_registry = {
        'grid': GridSearchOptimizer,
        'random': RandomSearchOptimizer,
        'bayesian': BayesianOptimizer,
        'optuna': OptunaOptimizer
    }
    
    @classmethod
    def create_optimizer(cls, optimizer_type: str, config: OptimizationConfig) -> BaseOptimizer:
        if optimizer_type not in cls._optimizer_registry:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        optimizer_class = cls._optimizer_registry[optimizer_type]
        return optimizer_class(config)
    
    @classmethod
    def create_grid_search(cls, search_space: Dict[str, Any], **kwargs) -> GridSearchOptimizer:
        config = OptimizationConfig(search_space=search_space, **kwargs)
        return GridSearchOptimizer(config)
    
    @classmethod
    def create_random_search(cls, search_space: Dict[str, Any], **kwargs) -> RandomSearchOptimizer:
        config = OptimizationConfig(search_space=search_space, **kwargs)
        return RandomSearchOptimizer(config)
    
    @classmethod
    def create_bayesian_search(cls, search_space: Dict[str, Any], **kwargs) -> BayesianOptimizer:
        config = OptimizationConfig(search_space=search_space, **kwargs)
        return BayesianOptimizer(config)
    
    @classmethod
    def create_optuna_search(cls, search_space: Dict[str, Any], **kwargs) -> OptunaOptimizer:
        config = OptimizationConfig(search_space=search_space, **kwargs)
        return OptunaOptimizer(config)
    
    @classmethod
    def get_available_optimizers(cls) -> List[str]:
        return list(cls._optimizer_registry.keys())
    
    @classmethod
    def get_best_optimizer(cls, n_trials: int = 100) -> str:
        if n_trials <= 20:
            return 'grid'
        elif n_trials <= 100:
            return 'random'
        else:
            return 'optuna'
    
    @classmethod
    def create_auto_optimizer(cls, search_space: Dict[str, Any], 
                             n_trials: int = 100, **kwargs) -> BaseOptimizer:
        optimizer_type = cls.get_best_optimizer(n_trials)
        config = OptimizationConfig(search_space=search_space, n_trials=n_trials, **kwargs)
        return cls.create_optimizer(optimizer_type, config)
    
    @classmethod
    def register_optimizer(cls, name: str, optimizer_class: type):
        cls._optimizer_registry[name] = optimizer_class


def create_quick_optimizer(model_name: str, target_type: str = 'regression', 
                          optimizer_type: str = 'optuna', n_trials: int = 100) -> BaseOptimizer:
    
    if model_name == 'xgboost':
        if target_type == 'classification':
            search_space = {
                'n_estimators': {'type': 'int', 'low': 100, 'high': 1000},
                'max_depth': {'type': 'int', 'low': 3, 'high': 8},
                'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
                'subsample': {'type': 'uniform', 'low': 0.7, 'high': 1.0},
                'colsample_bytree': {'type': 'uniform', 'low': 0.7, 'high': 1.0}
            }
            scoring = 'f1_weighted'
        else:
            search_space = {
                'n_estimators': {'type': 'int', 'low': 100, 'high': 1000},
                'max_depth': {'type': 'int', 'low': 3, 'high': 8},
                'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
                'subsample': {'type': 'uniform', 'low': 0.7, 'high': 1.0},
                'colsample_bytree': {'type': 'uniform', 'low': 0.7, 'high': 1.0}
            }
            scoring = 'neg_mean_squared_error'
    
    elif model_name == 'lightgbm':
        if target_type == 'classification':
            search_space = {
                'n_estimators': {'type': 'int', 'low': 100, 'high': 1000},
                'max_depth': {'type': 'int', 'low': 3, 'high': 8},
                'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
                'subsample': {'type': 'uniform', 'low': 0.7, 'high': 1.0},
                'colsample_bytree': {'type': 'uniform', 'low': 0.7, 'high': 1.0},
                'num_leaves': {'type': 'int', 'low': 10, 'high': 200}
            }
            scoring = 'f1_weighted'
        else:
            search_space = {
                'n_estimators': {'type': 'int', 'low': 100, 'high': 1000},
                'max_depth': {'type': 'int', 'low': 3, 'high': 8},
                'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
                'subsample': {'type': 'uniform', 'low': 0.7, 'high': 1.0},
                'colsample_bytree': {'type': 'uniform', 'low': 0.7, 'high': 1.0},
                'num_leaves': {'type': 'int', 'low': 10, 'high': 200}
            }
            scoring = 'neg_mean_squared_error'
    
    elif model_name == 'catboost':
        if target_type == 'classification':
            search_space = {
                'iterations': {'type': 'int', 'low': 100, 'high': 1000},
                'depth': {'type': 'int', 'low': 3, 'high': 8},
                'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
                'l2_leaf_reg': {'type': 'loguniform', 'low': 0.1, 'high': 10}
            }
            scoring = 'f1_weighted'
        else:
            search_space = {
                'iterations': {'type': 'int', 'low': 100, 'high': 1000},
                'depth': {'type': 'int', 'low': 3, 'high': 8},
                'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
                'l2_leaf_reg': {'type': 'loguniform', 'low': 0.1, 'high': 10}
            }
            scoring = 'neg_mean_squared_error'
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    config = OptimizationConfig(
        search_space=search_space,
        n_trials=n_trials,
        cv_folds=5,
        scoring=scoring,
        direction='maximize' if target_type == 'classification' else 'maximize'
    )
    
    return OptimizerFactory.create_optimizer(optimizer_type, config)