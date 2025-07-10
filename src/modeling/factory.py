from typing import Dict, Any, Optional, List
from .base import BaseModel, ModelConfig
from .tree_models import (
    XGBoostModel, LightGBMModel, CatBoostModel,
    create_xgboost_config, create_lightgbm_config, create_catboost_config
)
from .linear_models import (
    LinearModel, RidgeModel, LassoModel, ElasticNetModel, LogisticModel,
    create_linear_config, create_ridge_config, create_lasso_config
)


class ModelFactory:
    _model_registry = {
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'catboost': CatBoostModel,
        'linear': LinearModel,
        'ridge': RidgeModel,
        'lasso': LassoModel,
        'elasticnet': ElasticNetModel,
        'logistic': LogisticModel
    }
    
    _config_registry = {
        'xgboost': create_xgboost_config,
        'lightgbm': create_lightgbm_config,
        'catboost': create_catboost_config,
        'linear': create_linear_config,
        'ridge': create_ridge_config,
        'lasso': create_lasso_config
    }
    
    @classmethod
    def create_model(cls, config: ModelConfig) -> BaseModel:
        if config.model_type not in cls._model_registry:
            raise ValueError(f"Unknown model type: {config.model_type}")
            
        model_class = cls._model_registry[config.model_type]
        return model_class(config)
    
    @classmethod
    def create_model_from_name(cls, model_name: str, 
                             target_type: str = 'regression',
                             **params) -> BaseModel:
        if model_name not in cls._config_registry:
            raise ValueError(f"Unknown model name: {model_name}")
            
        config_func = cls._config_registry[model_name]
        config = config_func(target_type=target_type, **params)
        return cls.create_model(config)
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        return list(cls._model_registry.keys())
    
    @classmethod
    def get_default_models(cls, target_type: str = 'regression') -> List[BaseModel]:
        models = []
        
        xgb_config = create_xgboost_config(target_type=target_type)
        models.append(cls.create_model(xgb_config))
        
        lgb_config = create_lightgbm_config(target_type=target_type)
        models.append(cls.create_model(lgb_config))
        
        cb_config = create_catboost_config(target_type=target_type)
        models.append(cls.create_model(cb_config))
        
        if target_type == 'classification':
            logistic_config = create_ridge_config(target_type='classification')
            logistic_config.name = 'logistic_ridge'
            models.append(cls.create_model(logistic_config))
        else:
            ridge_config = create_ridge_config(target_type='regression')
            models.append(cls.create_model(ridge_config))
            
        return models
    
    @classmethod
    def create_ensemble_models(cls, target_type: str = 'regression') -> List[BaseModel]:
        models = []
        
        xgb_configs = [
            create_xgboost_config(target_type=target_type, learning_rate=0.05, max_depth=4),
            create_xgboost_config(target_type=target_type, learning_rate=0.1, max_depth=6),
            create_xgboost_config(target_type=target_type, learning_rate=0.2, max_depth=8)
        ]
        
        lgb_configs = [
            create_lightgbm_config(target_type=target_type, learning_rate=0.05, max_depth=4),
            create_lightgbm_config(target_type=target_type, learning_rate=0.1, max_depth=6),
            create_lightgbm_config(target_type=target_type, learning_rate=0.2, max_depth=8)
        ]
        
        cb_configs = [
            create_catboost_config(target_type=target_type, learning_rate=0.05, depth=4),
            create_catboost_config(target_type=target_type, learning_rate=0.1, depth=6),
            create_catboost_config(target_type=target_type, learning_rate=0.2, depth=8)
        ]
        
        for i, config in enumerate(xgb_configs):
            config.name = f'xgboost_{i+1}'
            models.append(cls.create_model(config))
            
        for i, config in enumerate(lgb_configs):
            config.name = f'lightgbm_{i+1}'
            models.append(cls.create_model(config))
            
        for i, config in enumerate(cb_configs):
            config.name = f'catboost_{i+1}'
            models.append(cls.create_model(config))
            
        return models
    
    @classmethod
    def register_model(cls, name: str, model_class: type):
        cls._model_registry[name] = model_class
    
    @classmethod
    def register_config(cls, name: str, config_func: callable):
        cls._config_registry[name] = config_func
    
    @classmethod
    def create_ensemble_model(cls, ensemble_type: str, target_type: str = 'regression', 
                              base_models: list = None, **kwargs):
        """Create ensemble model"""
        try:
            from .ensemble import VotingEnsemble, StackingEnsemble
            
            if base_models is None:
                # Create default base models
                base_models = [
                    cls.create_model_from_name('xgboost', target_type=target_type, n_estimators=100),
                    cls.create_model_from_name('lightgbm', target_type=target_type, n_estimators=100),
                ]
            
            if ensemble_type.lower() == 'voting':
                return VotingEnsemble(base_models, target_type=target_type, **kwargs)
            elif ensemble_type.lower() == 'stacking':
                return StackingEnsemble(base_models, target_type=target_type, **kwargs)
            else:
                raise ValueError(f"Unknown ensemble type: {ensemble_type}")
                
        except ImportError:
            # If ensemble module is not available, skip
            raise NotImplementedError(f"Ensemble model '{ensemble_type}' not available")


def create_kaggle_models(target_type: str = 'regression') -> List[BaseModel]:
    factory = ModelFactory()
    
    models = []
    
    xgb_1 = factory.create_model_from_name(
        'xgboost', target_type=target_type,
        learning_rate=0.05, max_depth=4, n_estimators=1500,
        subsample=0.8, colsample_bytree=0.8
    )
    xgb_1.config.name = 'xgb_conservative'
    models.append(xgb_1)
    
    xgb_2 = factory.create_model_from_name(
        'xgboost', target_type=target_type,
        learning_rate=0.1, max_depth=6, n_estimators=1000,
        subsample=0.9, colsample_bytree=0.9
    )
    xgb_2.config.name = 'xgb_balanced'
    models.append(xgb_2)
    
    lgb_1 = factory.create_model_from_name(
        'lightgbm', target_type=target_type,
        learning_rate=0.05, max_depth=4, n_estimators=1500,
        subsample=0.8, colsample_bytree=0.8
    )
    lgb_1.config.name = 'lgb_conservative'
    models.append(lgb_1)
    
    lgb_2 = factory.create_model_from_name(
        'lightgbm', target_type=target_type,
        learning_rate=0.1, max_depth=6, n_estimators=1000,
        subsample=0.9, colsample_bytree=0.9
    )
    lgb_2.config.name = 'lgb_balanced'
    models.append(lgb_2)
    
    cb_1 = factory.create_model_from_name(
        'catboost', target_type=target_type,
        learning_rate=0.05, depth=4, iterations=1500
    )
    cb_1.config.name = 'cb_conservative'
    models.append(cb_1)
    
    cb_2 = factory.create_model_from_name(
        'catboost', target_type=target_type,
        learning_rate=0.1, depth=6, iterations=1000
    )
    cb_2.config.name = 'cb_balanced'
    models.append(cb_2)
    
    return models