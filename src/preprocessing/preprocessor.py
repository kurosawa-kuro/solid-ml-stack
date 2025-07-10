import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from sklearn.model_selection import train_test_split
from .pipeline import PreprocessingPipeline, CustomPreprocessingPipeline


class Preprocessor:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.pipeline = None
        self.feature_names = None
        self.target_name = None
        
    def prepare_data(self, 
                    df: pd.DataFrame,
                    target_col: str,
                    test_size: float = 0.2,
                    random_state: int = 42,
                    stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        self.feature_names = X.columns.tolist()
        self.target_name = target_col
        
        stratify_col = y if stratify and y.dtype in ['object', 'category'] else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_col
        )
        
        return X_train, X_test, y_train, y_test
        
    def create_pipeline(self, custom_steps: Optional[List[tuple]] = None) -> PreprocessingPipeline:
        if custom_steps:
            self.pipeline = PreprocessingPipeline(steps=custom_steps)
        elif self.config:
            self.pipeline = CustomPreprocessingPipeline(self.config)
        else:
            self.pipeline = PreprocessingPipeline()
            
        return self.pipeline
        
    def fit_transform(self, 
                     X_train: pd.DataFrame,
                     y_train: Optional[pd.Series] = None) -> pd.DataFrame:
        
        if self.pipeline is None:
            self.create_pipeline()
            
        return self.pipeline.fit_transform(X_train, y_train)
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None or not self.pipeline.fitted:
            raise ValueError("Pipeline must be fitted before transform")
            
        return self.pipeline.transform(X)
        
    def process_train_test(self,
                          X_train: pd.DataFrame,
                          X_test: pd.DataFrame,
                          y_train: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        X_train_processed = self.fit_transform(X_train, y_train)
        X_test_processed = self.transform(X_test)
        
        return X_train_processed, X_test_processed
        
    def get_feature_names_out(self) -> List[str]:
        if self.pipeline is None or not self.pipeline.fitted:
            raise ValueError("Pipeline must be fitted first")
            
        last_step = self.pipeline.pipeline.steps[-1][1]
        if hasattr(last_step, 'get_feature_names_out'):
            return last_step.get_feature_names_out()
        elif hasattr(last_step, 'selected_features'):
            return last_step.selected_features
        else:
            return self.feature_names
            
    def save(self, filepath: str):
        if self.pipeline:
            self.pipeline.save_pipeline(filepath)
            
    def load(self, filepath: str):
        if self.pipeline is None:
            self.pipeline = PreprocessingPipeline()
        self.pipeline.load_pipeline(filepath)
        
    @staticmethod
    def quick_preprocess(df: pd.DataFrame,
                        target_col: str,
                        config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        
        preprocessor = Preprocessor(config)
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(df, target_col)
        X_train_processed, X_test_processed = preprocessor.process_train_test(
            X_train, X_test, y_train
        )
        
        return {
            'X_train': X_train_processed,
            'X_test': X_test_processed,
            'y_train': y_train,
            'y_test': y_test,
            'preprocessor': preprocessor,
            'feature_names': preprocessor.get_feature_names_out()
        }