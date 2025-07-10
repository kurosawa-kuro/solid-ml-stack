import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import datetime
import os


class SubmissionGenerator:
    def __init__(self, output_dir: str = "submissions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def create_submission(self, 
                         predictions: np.ndarray,
                         test_ids: Optional[np.ndarray] = None,
                         target_column: str = 'target',
                         id_column: str = 'id',
                         filename: Optional[str] = None) -> str:
        
        if test_ids is None:
            test_ids = np.arange(len(predictions))
        
        submission_df = pd.DataFrame({
            id_column: test_ids,
            target_column: predictions
        })
        
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"submission_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        submission_df.to_csv(filepath, index=False)
        
        print(f"Submission saved to: {filepath}")
        print(f"Submission shape: {submission_df.shape}")
        print(f"First few rows:")
        print(submission_df.head())
        
        return str(filepath)
    
    def create_ensemble_submission(self,
                                 predictions_dict: Dict[str, np.ndarray],
                                 weights: Optional[List[float]] = None,
                                 test_ids: Optional[np.ndarray] = None,
                                 target_column: str = 'target',
                                 id_column: str = 'id',
                                 filename: Optional[str] = None) -> str:
        
        if weights is None:
            weights = [1.0 / len(predictions_dict)] * len(predictions_dict)
        
        if len(weights) != len(predictions_dict):
            raise ValueError("Number of weights must match number of predictions")
        
        predictions_list = list(predictions_dict.values())
        predictions_array = np.array(predictions_list)
        
        ensemble_predictions = np.average(predictions_array, axis=0, weights=weights)
        
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ensemble_submission_{timestamp}.csv"
        
        filepath = self.create_submission(
            ensemble_predictions, test_ids, target_column, id_column, filename
        )
        
        print(f"\nEnsemble weights used:")
        for model_name, weight in zip(predictions_dict.keys(), weights):
            print(f"  {model_name}: {weight:.4f}")
        
        return filepath
    
    def create_rank_average_submission(self,
                                     predictions_dict: Dict[str, np.ndarray],
                                     test_ids: Optional[np.ndarray] = None,
                                     target_column: str = 'target',
                                     id_column: str = 'id',
                                     filename: Optional[str] = None) -> str:
        
        predictions_df = pd.DataFrame(predictions_dict)
        
        rank_df = predictions_df.rank(method='average')
        
        avg_ranks = rank_df.mean(axis=1)
        
        rank_predictions = avg_ranks.values
        
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rank_average_submission_{timestamp}.csv"
        
        filepath = self.create_submission(
            rank_predictions, test_ids, target_column, id_column, filename
        )
        
        print(f"\nRank averaging used for {len(predictions_dict)} models")
        
        return filepath
    
    def create_geometric_mean_submission(self,
                                       predictions_dict: Dict[str, np.ndarray],
                                       test_ids: Optional[np.ndarray] = None,
                                       target_column: str = 'target',
                                       id_column: str = 'id',
                                       filename: Optional[str] = None) -> str:
        
        predictions_list = list(predictions_dict.values())
        predictions_array = np.array(predictions_list)
        
        predictions_array = np.maximum(predictions_array, 1e-8)
        
        geometric_mean = np.exp(np.mean(np.log(predictions_array), axis=0))
        
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"geometric_mean_submission_{timestamp}.csv"
        
        filepath = self.create_submission(
            geometric_mean, test_ids, target_column, id_column, filename
        )
        
        print(f"\nGeometric mean used for {len(predictions_dict)} models")
        
        return filepath
    
    def create_multiple_submissions(self,
                                  models_predictions: Dict[str, np.ndarray],
                                  test_ids: Optional[np.ndarray] = None,
                                  target_column: str = 'target',
                                  id_column: str = 'id') -> List[str]:
        
        filepaths = []
        
        for model_name, predictions in models_predictions.items():
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_submission_{timestamp}.csv"
            
            filepath = self.create_submission(
                predictions, test_ids, target_column, id_column, filename
            )
            filepaths.append(filepath)
        
        return filepaths
    
    def validate_submission(self, 
                          submission_path: str,
                          expected_rows: Optional[int] = None,
                          expected_columns: Optional[List[str]] = None) -> bool:
        
        try:
            df = pd.read_csv(submission_path)
            
            print(f"Validation for: {submission_path}")
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            
            if expected_rows and df.shape[0] != expected_rows:
                print(f"WARNING: Expected {expected_rows} rows, got {df.shape[0]}")
                return False
            
            if expected_columns:
                missing_cols = set(expected_columns) - set(df.columns)
                if missing_cols:
                    print(f"WARNING: Missing columns: {missing_cols}")
                    return False
            
            if df.isnull().any().any():
                print("WARNING: Submission contains null values")
                print(df.isnull().sum())
                return False
            
            print("Validation passed!")
            return True
            
        except Exception as e:
            print(f"Validation failed: {e}")
            return False
    
    def get_submission_summary(self, submission_path: str) -> Dict[str, Any]:
        df = pd.read_csv(submission_path)
        
        summary = {
            'filename': Path(submission_path).name,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'has_nulls': df.isnull().any().any(),
            'file_size_mb': os.path.getsize(submission_path) / (1024 * 1024)
        }
        
        if df.shape[1] >= 2:
            target_col = df.columns[1]
            summary['target_stats'] = {
                'mean': df[target_col].mean(),
                'std': df[target_col].std(),
                'min': df[target_col].min(),
                'max': df[target_col].max(),
                'unique_values': df[target_col].nunique()
            }
        
        return summary