import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import warnings

warnings.filterwarnings('ignore')


class NumericScaler(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = 'standard', columns: Optional[List[str]] = None):
        self.method = method
        self.columns = columns
        self.scalers = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()
            
        for col in self.columns:
            if col in X.columns:
                if self.method == 'standard':
                    scaler = StandardScaler()
                elif self.method == 'minmax':
                    scaler = MinMaxScaler()
                elif self.method == 'robust':
                    scaler = RobustScaler()
                else:
                    raise ValueError(f"Unknown scaling method: {self.method}")
                    
                scaler.fit(X[[col]])
                self.scalers[col] = scaler
                
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        for col, scaler in self.scalers.items():
            if col in X_copy.columns:
                X_copy[col] = scaler.transform(X_copy[[col]])
        return X_copy


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = 'label', columns: Optional[List[str]] = None,
                 handle_unknown: str = 'use_encoded_value', unknown_value: int = -1):
        self.method = method
        self.columns = columns
        self.encoders = {}
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        
    def fit(self, X: pd.DataFrame, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
        for col in self.columns:
            if col in X.columns:
                if self.method == 'label':
                    encoder = LabelEncoder()
                    encoder.fit(X[col].fillna('missing'))
                    self.encoders[col] = encoder
                elif self.method == 'onehot':
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoder.fit(X[[col]])
                    self.encoders[col] = encoder
                elif self.method == 'target':
                    if y is None:
                        raise ValueError("Target encoding requires y")
                    mapping = X.groupby(col)[y.name].mean().to_dict()
                    self.encoders[col] = mapping
                    
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        
        for col, encoder in self.encoders.items():
            if col in X_copy.columns:
                if self.method == 'label':
                    X_copy[col] = X_copy[col].fillna('missing')
                    X_copy[col] = X_copy[col].apply(
                        lambda x: encoder.transform([x])[0] 
                        if x in encoder.classes_ else self.unknown_value
                    )
                elif self.method == 'onehot':
                    encoded = encoder.transform(X_copy[[col]])
                    feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X_copy.index)
                    X_copy = pd.concat([X_copy.drop(columns=[col]), encoded_df], axis=1)
                elif self.method == 'target':
                    X_copy[col] = X_copy[col].map(encoder).fillna(encoder.get('missing', 0))
                    
        return X_copy


class MissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_strategy: str = 'mean', categorical_strategy: str = 'mode',
                 numeric_cols: Optional[List[str]] = None, categorical_cols: Optional[List[str]] = None):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.imputers = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        if self.numeric_cols is None:
            self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if self.categorical_cols is None:
            self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
        for col in self.numeric_cols:
            if col in X.columns and X[col].isnull().any():
                if self.numeric_strategy == 'knn':
                    imputer = KNNImputer(n_neighbors=5)
                else:
                    imputer = SimpleImputer(strategy=self.numeric_strategy)
                imputer.fit(X[[col]])
                self.imputers[col] = imputer
                
        for col in self.categorical_cols:
            if col in X.columns and X[col].isnull().any():
                if self.categorical_strategy == 'mode':
                    mode_value = X[col].mode()[0] if not X[col].mode().empty else 'missing'
                    self.imputers[col] = mode_value
                else:
                    self.imputers[col] = 'missing'
                    
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        
        for col, imputer in self.imputers.items():
            if col in X_copy.columns:
                if isinstance(imputer, (SimpleImputer, KNNImputer)):
                    X_copy[col] = imputer.transform(X_copy[[col]])
                else:
                    X_copy[col] = X_copy[col].fillna(imputer)
                    
        return X_copy


class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = 'iqr', threshold: float = 1.5, 
                 columns: Optional[List[str]] = None):
        self.method = method
        self.threshold = threshold
        self.columns = columns
        self.bounds = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()
            
        for col in self.columns:
            if col in X.columns:
                if self.method == 'iqr':
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.threshold * IQR
                    upper_bound = Q3 + self.threshold * IQR
                elif self.method == 'zscore':
                    mean = X[col].mean()
                    std = X[col].std()
                    lower_bound = mean - self.threshold * std
                    upper_bound = mean + self.threshold * std
                else:
                    raise ValueError(f"Unknown outlier method: {self.method}")
                    
                self.bounds[col] = (lower_bound, upper_bound)
                
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        
        for col, (lower, upper) in self.bounds.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].clip(lower=lower, upper=upper)
                
        return X_copy


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = 'variance', threshold: float = 0.01,
                 n_features: Optional[int] = None):
        self.method = method
        self.threshold = threshold
        self.n_features = n_features
        self.selected_features = []
        
    def fit(self, X: pd.DataFrame, y=None):
        if self.method == 'variance':
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            variances = X[numeric_cols].var()
            self.selected_features = variances[variances > self.threshold].index.tolist()
            
        elif self.method == 'correlation' and y is not None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            correlations = pd.DataFrame({
                col: [abs(X[col].corr(y))] for col in numeric_cols
            }).T
            correlations.columns = ['correlation']
            correlations = correlations.sort_values('correlation', ascending=False)
            
            if self.n_features:
                self.selected_features = correlations.head(self.n_features).index.tolist()
            else:
                self.selected_features = correlations[
                    correlations['correlation'] > self.threshold
                ].index.tolist()
                
        else:
            self.selected_features = X.columns.tolist()
            
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        self.selected_features.extend(categorical_cols.tolist())
        
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Only select features that exist in the current DataFrame
        available_features = [col for col in self.selected_features if col in X.columns]
        return X[available_features]