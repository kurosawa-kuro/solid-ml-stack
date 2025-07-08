"""
Standardized cross-validation implementation for Kaggle competitions.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

from src.utils.config import Config
from src.ml.models.model_factory import ModelFactory


logger = logging.getLogger(__name__)


class KaggleCrossValidator:
    """
    Standardized cross-validation for Kaggle competitions.
    """
    
    def __init__(self, config: Config, random_state: int = 42):
        self.config = config
        self.random_state = random_state
        
    def get_cv_splitter(
        self,
        cv_type: str = "kfold",
        n_splits: int = 5,
        shuffle: bool = True,
        target: Optional[np.ndarray] = None
    ):
        """
        Get cross-validation splitter based on type.
        
        Args:
            cv_type: Type of CV ('kfold', 'stratified', 'timeseries')
            n_splits: Number of splits
            shuffle: Whether to shuffle data
            target: Target values (required for stratified CV)
            
        Returns:
            Cross-validation splitter
        """
        if cv_type == "kfold":
            return KFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_state)
        elif cv_type == "stratified":
            if target is None:
                raise ValueError("Target values required for stratified CV")
            return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_state)
        elif cv_type == "timeseries":
            return TimeSeriesSplit(n_splits=n_splits)
        else:
            raise ValueError(f"Unknown CV type: {cv_type}")
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        problem_type: str = "regression"
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics based on problem type.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            problem_type: Type of problem ('regression', 'classification')
            
        Returns:
            Dictionary of metrics
        """
        if problem_type == "regression":
            return {
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "mae": mean_absolute_error(y_true, y_pred),
                "r2": r2_score(y_true, y_pred)
            }
        elif problem_type == "classification":
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="weighted"),
                "recall": recall_score(y_true, y_pred, average="weighted"),
                "f1": f1_score(y_true, y_pred, average="weighted")
            }
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
    
    def cross_validate_model(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        cv_type: str = "kfold",
        n_splits: int = 5,
        problem_type: str = "regression",
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Perform cross-validation for a single model.
        
        Args:
            model_name: Name of the model
            X: Feature matrix
            y: Target values
            cv_type: Type of cross-validation
            n_splits: Number of CV folds
            problem_type: Type of problem
            return_predictions: Whether to return out-of-fold predictions
            
        Returns:
            Dictionary containing CV results
        """
        # Get CV splitter
        cv_splitter = self.get_cv_splitter(cv_type, n_splits, target=y.values)
        
        # Initialize results storage
        fold_metrics = []
        oof_predictions = np.zeros(len(X))
        
        logger.info(f"Starting {n_splits}-fold cross-validation for {model_name}")
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
            logger.info(f"Training fold {fold + 1}/{n_splits}")
            
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create and train model
            model = ModelFactory.create_model(model_name, self.config)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Store out-of-fold predictions
            oof_predictions[val_idx] = y_pred
            
            # Calculate metrics
            fold_metrics.append(self.calculate_metrics(y_val.values, y_pred, problem_type))
            
            logger.info(f"Fold {fold + 1} metrics: {fold_metrics[-1]}")
        
        # Calculate overall metrics
        overall_metrics = self.calculate_metrics(y.values, oof_predictions, problem_type)
        
        # Aggregate fold metrics
        aggregated_metrics = {}
        for metric_name in fold_metrics[0].keys():
            values = [fold[metric_name] for fold in fold_metrics]
            aggregated_metrics[f"{metric_name}_mean"] = np.mean(values)
            aggregated_metrics[f"{metric_name}_std"] = np.std(values)
        
        results = {
            "model_name": model_name,
            "cv_type": cv_type,
            "n_splits": n_splits,
            "fold_metrics": fold_metrics,
            "aggregated_metrics": aggregated_metrics,
            "overall_metrics": overall_metrics,
            "oof_score": overall_metrics.get("rmse", overall_metrics.get("accuracy", 0))
        }
        
        if return_predictions:
            results["oof_predictions"] = oof_predictions
            
        logger.info(f"CV completed for {model_name}. Overall score: {results['oof_score']:.4f}")
        
        return results
    
    def cross_validate_multiple_models(
        self,
        model_names: List[str],
        X: pd.DataFrame,
        y: pd.Series,
        cv_type: str = "kfold",
        n_splits: int = 5,
        problem_type: str = "regression"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform cross-validation for multiple models.
        
        Args:
            model_names: List of model names
            X: Feature matrix
            y: Target values
            cv_type: Type of cross-validation
            n_splits: Number of CV folds
            problem_type: Type of problem
            
        Returns:
            Dictionary mapping model names to their CV results
        """
        results = {}
        
        for model_name in model_names:
            try:
                logger.info(f"Starting CV for {model_name}")
                results[model_name] = self.cross_validate_model(
                    model_name, X, y, cv_type, n_splits, problem_type
                )
            except Exception as e:
                logger.error(f"CV failed for {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    def generate_stacking_features(
        self,
        base_model_names: List[str],
        X: pd.DataFrame,
        y: pd.Series,
        cv_type: str = "kfold",
        n_splits: int = 5
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate out-of-fold predictions for stacking.
        
        Args:
            base_model_names: List of base model names
            X: Feature matrix
            y: Target values
            cv_type: Type of cross-validation
            n_splits: Number of CV folds
            
        Returns:
            Tuple of (stacking features, individual model predictions)
        """
        # Get CV splitter
        cv_splitter = self.get_cv_splitter(cv_type, n_splits, target=y.values)
        
        # Initialize storage
        stacking_features = np.zeros((len(X), len(base_model_names)))
        model_predictions = {model_name: np.zeros(len(X)) for model_name in base_model_names}
        
        logger.info(f"Generating stacking features using {n_splits}-fold CV")
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
            logger.info(f"Processing fold {fold + 1}/{n_splits}")
            
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train and predict for each base model
            for i, model_name in enumerate(base_model_names):
                try:
                    # Create and train model
                    model = ModelFactory.create_model(model_name, self.config)
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_val)
                    
                    # Store predictions
                    stacking_features[val_idx, i] = y_pred
                    model_predictions[model_name][val_idx] = y_pred
                    
                except Exception as e:
                    logger.warning(f"Failed to train {model_name} in fold {fold + 1}: {e}")
                    # Use zeros for failed models
                    stacking_features[val_idx, i] = 0
                    model_predictions[model_name][val_idx] = 0
        
        return stacking_features, model_predictions
    
    def cv_ensemble_evaluation(
        self,
        base_model_names: List[str],
        X: pd.DataFrame,
        y: pd.Series,
        ensemble_methods: List[str] = ["average", "weighted", "median"],
        cv_type: str = "kfold",
        n_splits: int = 5,
        problem_type: str = "regression"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate ensemble methods using cross-validation.
        
        Args:
            base_model_names: List of base model names
            X: Feature matrix
            y: Target values
            ensemble_methods: List of ensemble methods to evaluate
            cv_type: Type of cross-validation
            n_splits: Number of CV folds
            problem_type: Type of problem
            
        Returns:
            Dictionary mapping ensemble methods to their CV results
        """
        # Generate stacking features
        stacking_features, model_predictions = self.generate_stacking_features(
            base_model_names, X, y, cv_type, n_splits
        )
        
        results = {}
        
        # Evaluate each ensemble method
        for method in ensemble_methods:
            try:
                if method == "average":
                    ensemble_pred = np.mean(stacking_features, axis=1)
                elif method == "weighted":
                    # Use equal weights for simplicity
                    weights = np.ones(len(base_model_names)) / len(base_model_names)
                    ensemble_pred = np.average(stacking_features, axis=1, weights=weights)
                elif method == "median":
                    ensemble_pred = np.median(stacking_features, axis=1)
                else:
                    logger.warning(f"Unknown ensemble method: {method}")
                    continue
                
                # Calculate metrics
                metrics = self.calculate_metrics(y.values, ensemble_pred, problem_type)
                
                results[f"ensemble_{method}"] = {
                    "method": method,
                    "base_models": base_model_names,
                    "metrics": metrics,
                    "oof_score": metrics.get("rmse", metrics.get("accuracy", 0))
                }
                
                logger.info(f"Ensemble {method} score: {results[f'ensemble_{method}']['oof_score']:.4f}")
                
            except Exception as e:
                logger.error(f"Ensemble {method} evaluation failed: {e}")
                results[f"ensemble_{method}"] = {"error": str(e)}
        
        return results