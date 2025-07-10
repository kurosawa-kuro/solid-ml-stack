from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class KaggleConfig:
    problem_type: str = 'regression'
    target_column: str = 'target'
    id_column: str = 'id'
    
    preprocessing: Dict[str, Any] = None
    feature_engineering: Dict[str, Any] = None
    modeling: Dict[str, Any] = None
    optimization: Dict[str, Any] = None
    ensemble: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.preprocessing is None:
            self.preprocessing = {
                'handle_missing': True,
                'handle_outliers': True,
                'encode_categorical': True,
                'scale_numeric': True,
                'numeric_impute_strategy': 'mean',
                'categorical_impute_strategy': 'mode',
                'outlier_method': 'iqr',
                'outlier_threshold': 1.5,
                'encoding_method': 'label',
                'scaling_method': 'standard'
            }
        
        if self.feature_engineering is None:
            self.feature_engineering = {
                'numeric_features': True,
                'categorical_features': True,
                'interaction_features': True,
                'polynomial_features': False,
                'target_encoding': self.problem_type == 'classification',
                'datetime_features': True,
                'aggregation_features': False,
                'numeric_transformations': ['log', 'sqrt', 'square'],
                'min_frequency': 10,
                'max_interactions': 50
            }
        
        if self.modeling is None:
            self.modeling = {
                'models': ['xgboost', 'lightgbm', 'catboost'],
                'use_ensemble': True,
                'ensemble_methods': ['average', 'weighted', 'stacking']
            }
        
        if self.optimization is None:
            self.optimization = {
                'method': 'optuna',
                'n_trials': 100,
                'cv_folds': 5,
                'timeout': None,
                'early_stopping_rounds': 20
            }
        
        if self.ensemble is None:
            self.ensemble = {
                'use_ensemble': True,
                'methods': ['average', 'weighted'],
                'meta_model': 'ridge'
            }


@dataclass
class CompetitionConfig:
    name: str
    type: str
    metric: str
    train_path: str
    test_path: str
    sample_submission_path: Optional[str] = None
    
    kaggle_config: KaggleConfig = None
    
    def __post_init__(self):
        if self.kaggle_config is None:
            self.kaggle_config = KaggleConfig(problem_type=self.type)


class ConfigPresets:
    @staticmethod
    def regression_competition() -> KaggleConfig:
        return KaggleConfig(
            problem_type='regression',
            preprocessing={
                'handle_missing': True,
                'handle_outliers': True,
                'encode_categorical': True,
                'scale_numeric': True,
                'outlier_method': 'iqr',
                'outlier_threshold': 2.0
            },
            feature_engineering={
                'numeric_features': True,
                'categorical_features': True,
                'interaction_features': True,
                'polynomial_features': True,
                'target_encoding': False,
                'max_interactions': 100
            },
            optimization={
                'method': 'optuna',
                'n_trials': 200,
                'cv_folds': 5
            }
        )
    
    @staticmethod
    def classification_competition() -> KaggleConfig:
        return KaggleConfig(
            problem_type='classification',
            preprocessing={
                'handle_missing': True,
                'handle_outliers': False,
                'encode_categorical': True,
                'scale_numeric': True,
                'encoding_method': 'target'
            },
            feature_engineering={
                'numeric_features': True,
                'categorical_features': True,
                'interaction_features': True,
                'target_encoding': True,
                'max_interactions': 50
            },
            optimization={
                'method': 'bayesian',
                'n_trials': 150,
                'cv_folds': 5
            }
        )
    
    @staticmethod
    def tabular_competition() -> KaggleConfig:
        return KaggleConfig(
            problem_type='regression',
            preprocessing={
                'handle_missing': True,
                'handle_outliers': True,
                'encode_categorical': True,
                'scale_numeric': False
            },
            feature_engineering={
                'numeric_features': True,
                'categorical_features': True,
                'interaction_features': True,
                'polynomial_features': False,
                'target_encoding': True,
                'aggregation_features': True
            },
            modeling={
                'models': ['xgboost', 'lightgbm', 'catboost'],
                'use_ensemble': True
            }
        )
    
    @staticmethod
    def time_series_competition() -> KaggleConfig:
        return KaggleConfig(
            problem_type='regression',
            feature_engineering={
                'numeric_features': True,
                'datetime_features': True,
                'lag_features': True,
                'window_features': True,
                'interaction_features': False
            }
        )


POPULAR_COMPETITIONS = {
    'house_prices': CompetitionConfig(
        name='House Prices',
        type='regression',
        metric='rmse',
        train_path='data/house_prices/train.csv',
        test_path='data/house_prices/test.csv',
        kaggle_config=ConfigPresets.regression_competition()
    ),
    
    'titanic': CompetitionConfig(
        name='Titanic',
        type='classification',
        metric='accuracy',
        train_path='data/titanic/train.csv',
        test_path='data/titanic/test.csv',
        kaggle_config=ConfigPresets.classification_competition()
    ),
    
    'tabular_playground': CompetitionConfig(
        name='Tabular Playground',
        type='regression',
        metric='rmse',
        train_path='data/tabular/train.csv',
        test_path='data/tabular/test.csv',
        kaggle_config=ConfigPresets.tabular_competition()
    )
}