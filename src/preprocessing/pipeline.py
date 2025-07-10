import pandas as pd
from typing import List, Dict, Any, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from .transformers import (
    NumericScaler, CategoricalEncoder, MissingValueHandler,
    OutlierHandler, FeatureSelector
)


class PreprocessingPipeline:
    def __init__(self, steps: Optional[List[tuple]] = None):
        self.steps = steps or self._get_default_steps()
        self.pipeline = None
        self.fitted = False
        
    def _get_default_steps(self) -> List[tuple]:
        return [
            ('missing_handler', MissingValueHandler()),
            ('outlier_handler', OutlierHandler()),
            ('categorical_encoder', CategoricalEncoder()),
            ('numeric_scaler', NumericScaler()),
            ('feature_selector', FeatureSelector())
        ]
        
    def add_step(self, name: str, transformer: BaseEstimator, position: Optional[int] = None):
        if position is None:
            self.steps.append((name, transformer))
        else:
            self.steps.insert(position, (name, transformer))
            
    def remove_step(self, name: str):
        self.steps = [(n, t) for n, t in self.steps if n != name]
        
    def build(self):
        self.pipeline = Pipeline(self.steps)
        return self
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if self.pipeline is None:
            self.build()
        self.pipeline.fit(X, y)
        self.fitted = True
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transform")
        return pd.DataFrame(
            self.pipeline.transform(X),
            index=X.index
        )
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)
        
    def get_params(self) -> Dict[str, Any]:
        if self.pipeline is None:
            return {}
        return self.pipeline.get_params()
        
    def set_params(self, **params):
        if self.pipeline is None:
            self.build()
        self.pipeline.set_params(**params)
        return self
        
    def save_pipeline(self, filepath: str):
        import joblib
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before saving")
        joblib.dump(self.pipeline, filepath)
        
    def load_pipeline(self, filepath: str):
        import joblib
        self.pipeline = joblib.load(filepath)
        self.fitted = True
        return self


class CustomPreprocessingPipeline(PreprocessingPipeline):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self._build_from_config()
        
    def _build_from_config(self):
        self.steps = []
        
        if self.config.get('handle_missing', True):
            self.steps.append((
                'missing_handler',
                MissingValueHandler(
                    numeric_strategy=self.config.get('numeric_impute_strategy', 'mean'),
                    categorical_strategy=self.config.get('categorical_impute_strategy', 'mode')
                )
            ))
            
        if self.config.get('handle_outliers', True):
            self.steps.append((
                'outlier_handler',
                OutlierHandler(
                    method=self.config.get('outlier_method', 'iqr'),
                    threshold=self.config.get('outlier_threshold', 1.5)
                )
            ))
            
        if self.config.get('encode_categorical', True):
            self.steps.append((
                'categorical_encoder',
                CategoricalEncoder(
                    method=self.config.get('encoding_method', 'label')
                )
            ))
            
        if self.config.get('scale_numeric', True):
            self.steps.append((
                'numeric_scaler',
                NumericScaler(
                    method=self.config.get('scaling_method', 'standard')
                )
            ))
            
        if self.config.get('select_features', False):
            self.steps.append((
                'feature_selector',
                FeatureSelector(
                    method=self.config.get('feature_selection_method', 'variance'),
                    threshold=self.config.get('feature_selection_threshold', 0.01)
                )
            ))