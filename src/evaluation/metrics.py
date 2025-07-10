import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, log_loss
)
import warnings
warnings.filterwarnings('ignore')


class MetricsCalculator:
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        metrics = {}
        
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        try:
            metrics['rmsle'] = np.sqrt(mean_squared_log_error(
                np.maximum(0, y_true), np.maximum(0, y_pred)
            ))
        except:
            metrics['rmsle'] = np.nan
        
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / np.maximum(1e-8, np.abs(y_true)))) * 100
        
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        
        return metrics
    
    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        metrics = {}
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        if len(np.unique(y_true)) == 2:
            metrics['precision'] = precision_score(y_true, y_pred, average='binary')
            metrics['recall'] = recall_score(y_true, y_pred, average='binary')
            metrics['f1'] = f1_score(y_true, y_pred, average='binary')
            
            if y_pred_proba is not None:
                if len(y_pred_proba.shape) > 1:
                    y_pred_proba = y_pred_proba[:, 1]
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
        else:
            metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
            metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
            
            metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
            
            if y_pred_proba is not None:
                try:
                    metrics['auc_ovr'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
                    metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                except:
                    pass
        
        return metrics
    
    @staticmethod
    def calculate_kaggle_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                               competition_type: str = 'regression') -> Dict[str, float]:
        
        if competition_type == 'regression':
            return MetricsCalculator.calculate_regression_metrics(y_true, y_pred)
        elif competition_type == 'classification':
            return MetricsCalculator.calculate_classification_metrics(y_true, y_pred)
        elif competition_type == 'binary_classification':
            return MetricsCalculator.calculate_classification_metrics(y_true, y_pred)
        else:
            raise ValueError(f"Unknown competition type: {competition_type}")
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
        print(f"\n{title}")
        print("=" * len(title))
        
        for metric_name, value in metrics.items():
            if np.isnan(value):
                print(f"{metric_name}: NaN")
            else:
                print(f"{metric_name}: {value:.4f}")
        print()
    
    @staticmethod
    def compare_models(results: Dict[str, Dict[str, float]], 
                      primary_metric: str = 'rmse') -> pd.DataFrame:
        
        df = pd.DataFrame(results).T
        
        if primary_metric in df.columns:
            ascending = primary_metric in ['mse', 'rmse', 'mae', 'log_loss', 'rmsle', 'mape']
            df = df.sort_values(primary_metric, ascending=ascending)
        
        return df


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return MetricsCalculator.calculate_regression_metrics(y_true, y_pred)


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                   y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    return MetricsCalculator.calculate_classification_metrics(y_true, y_pred, y_pred_proba)


def calculate_business_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                             cost_matrix: Optional[np.ndarray] = None) -> Dict[str, float]:
    
    metrics = {}
    
    if cost_matrix is not None:
        cm = confusion_matrix(y_true, y_pred)
        total_cost = np.sum(cm * cost_matrix)
        metrics['total_cost'] = total_cost
        metrics['avg_cost_per_sample'] = total_cost / len(y_true)
    
    residuals = y_true - y_pred
    metrics['mean_absolute_percentage_error'] = np.mean(np.abs(residuals / np.maximum(1e-8, np.abs(y_true)))) * 100
    
    percentiles = [90, 95, 99]
    for p in percentiles:
        metrics[f'percentile_{p}_error'] = np.percentile(np.abs(residuals), p)
    
    return metrics


class KaggleMetrics:
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(mean_squared_log_error(
            np.maximum(0, y_true), np.maximum(0, y_pred)
        ))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs((y_true - y_pred) / np.maximum(1e-8, np.abs(y_true)))) * 100
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return accuracy_score(y_true, y_pred)
    
    @staticmethod
    def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return f1_score(y_true, y_pred, average='weighted')
    
    @staticmethod
    def auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        if len(y_pred_proba.shape) > 1:
            y_pred_proba = y_pred_proba[:, 1]
        return roc_auc_score(y_true, y_pred_proba)
    
    @staticmethod
    def logloss(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        return log_loss(y_true, y_pred_proba)