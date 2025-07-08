"""
Advanced ensemble methods for Kaggle competitions.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize
import logging

from src.utils.config import Config
from src.ml.models.model_factory import ModelFactory
from src.kaggle.cross_validation import KaggleCrossValidator


logger = logging.getLogger(__name__)


class KaggleEnsemble:
    """
    Advanced ensemble methods specifically designed for Kaggle competitions.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.cv = KaggleCrossValidator(config)
        
    def blending_ensemble(
        self,
        base_model_names: List[str],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_blend: pd.DataFrame,
        y_blend: pd.Series,
        blend_method: str = "ridge"
    ) -> Dict[str, Any]:
        """
        Create blending ensemble using a holdout set.
        
        Args:
            base_model_names: List of base model names
            X_train: Training features
            y_train: Training targets
            X_blend: Blending features
            y_blend: Blending targets
            blend_method: Method for blending ('ridge', 'lasso', 'linear', 'rf')
            
        Returns:
            Dictionary containing the blending ensemble
        """
        logger.info(f"Creating blending ensemble with {len(base_model_names)} base models")
        
        # Train base models on training set
        base_models = {}
        blend_predictions = []
        
        for model_name in base_model_names:
            try:
                logger.info(f"Training base model: {model_name}")
                model = ModelFactory.create_model(model_name, self.config)
                model.fit(X_train, y_train)
                
                # Make predictions on blending set
                blend_pred = model.predict(X_blend)
                blend_predictions.append(blend_pred)
                base_models[model_name] = model
                
            except Exception as e:
                logger.warning(f"Failed to train {model_name}: {e}")
                continue
        
        if not base_models:
            raise ValueError("No base models were successfully trained")
        
        # Prepare blending features
        blend_features = np.column_stack(blend_predictions)
        
        # Train blending model
        if blend_method == "ridge":
            blender = Ridge(alpha=1.0)
        elif blend_method == "lasso":
            blender = Lasso(alpha=0.1)
        elif blend_method == "linear":
            blender = LinearRegression()
        elif blend_method == "rf":
            blender = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown blend method: {blend_method}")
        
        blender.fit(blend_features, y_blend)
        
        # Evaluate blending performance
        blend_pred_final = blender.predict(blend_features)
        blend_score = np.sqrt(mean_squared_error(y_blend, blend_pred_final))
        
        logger.info(f"Blending ensemble RMSE: {blend_score:.4f}")
        
        return {
            "base_models": base_models,
            "blender": blender,
            "blend_method": blend_method,
            "blend_score": blend_score,
            "base_model_names": list(base_models.keys())
        }
    
    def bayesian_ensemble(
        self,
        base_model_names: List[str],
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Create Bayesian ensemble using Bayesian optimization for weights.
        
        Args:
            base_model_names: List of base model names
            X: Feature matrix
            y: Target values
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary containing the Bayesian ensemble
        """
        logger.info(f"Creating Bayesian ensemble with {len(base_model_names)} base models")
        
        # Generate out-of-fold predictions
        stacking_features, model_predictions = self.cv.generate_stacking_features(
            base_model_names, X, y, cv_type="kfold", n_splits=cv_folds
        )
        
        # Define objective function for Bayesian optimization
        def objective(weights):
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize weights
            
            # Calculate weighted average prediction
            weighted_pred = np.average(stacking_features, axis=1, weights=weights)
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y, weighted_pred))
            return rmse
        
        # Initialize weights
        n_models = len(base_model_names)
        initial_weights = np.ones(n_models) / n_models
        
        # Constraints: weights must sum to 1 and be positive
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0.01, 1.0) for _ in range(n_models)]
        
        # Optimize weights
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        optimal_weights = result.x
        optimal_weights = optimal_weights / optimal_weights.sum()  # Normalize
        
        # Calculate final score
        final_pred = np.average(stacking_features, axis=1, weights=optimal_weights)
        final_score = np.sqrt(mean_squared_error(y, final_pred))
        
        logger.info(f"Bayesian ensemble RMSE: {final_score:.4f}")
        
        # Create weight mapping
        weight_mapping = dict(zip(base_model_names, optimal_weights))
        
        return {
            "base_model_names": base_model_names,
            "optimal_weights": optimal_weights,
            "weight_mapping": weight_mapping,
            "final_score": final_score,
            "optimization_result": result,
            "oof_predictions": model_predictions
        }
    
    def rank_ensemble(
        self,
        base_model_names: List[str],
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Create rank-based ensemble.
        
        Args:
            base_model_names: List of base model names
            X: Feature matrix
            y: Target values
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary containing the rank ensemble
        """
        logger.info(f"Creating rank ensemble with {len(base_model_names)} base models")
        
        # Generate out-of-fold predictions
        stacking_features, model_predictions = self.cv.generate_stacking_features(
            base_model_names, X, y, cv_type="kfold", n_splits=cv_folds
        )
        
        # Convert predictions to ranks
        rank_features = np.zeros_like(stacking_features)
        for i in range(stacking_features.shape[1]):
            rank_features[:, i] = pd.Series(stacking_features[:, i]).rank(pct=True)
        
        # Average the ranks
        rank_pred = np.mean(rank_features, axis=1)
        
        # Convert back to original scale by ranking and mapping
        rank_pred_sorted = np.argsort(rank_pred)
        y_sorted = np.sort(y)
        final_pred = np.zeros_like(rank_pred)
        final_pred[rank_pred_sorted] = y_sorted
        
        # Calculate score
        final_score = np.sqrt(mean_squared_error(y, final_pred))
        
        logger.info(f"Rank ensemble RMSE: {final_score:.4f}")
        
        return {
            "base_model_names": base_model_names,
            "rank_features": rank_features,
            "final_pred": final_pred,
            "final_score": final_score,
            "oof_predictions": model_predictions
        }
    
    def power_ensemble(
        self,
        base_model_names: List[str],
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        power_values: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Create power ensemble by raising predictions to different powers.
        
        Args:
            base_model_names: List of base model names
            X: Feature matrix
            y: Target values
            cv_folds: Number of CV folds
            power_values: List of power values to try
            
        Returns:
            Dictionary containing the power ensemble
        """
        if power_values is None:
            power_values = [0.5, 1.0, 1.5, 2.0]
        
        logger.info(f"Creating power ensemble with {len(base_model_names)} base models")
        
        # Generate out-of-fold predictions
        stacking_features, model_predictions = self.cv.generate_stacking_features(
            base_model_names, X, y, cv_type="kfold", n_splits=cv_folds
        )
        
        best_score = float('inf')
        best_powers = None
        best_pred = None
        
        # Try different power combinations
        for power in power_values:
            try:
                # Apply power transformation
                powered_features = np.power(np.abs(stacking_features), power) * np.sign(stacking_features)
                
                # Average the powered predictions
                power_pred = np.mean(powered_features, axis=1)
                
                # Calculate score
                score = np.sqrt(mean_squared_error(y, power_pred))
                
                if score < best_score:
                    best_score = score
                    best_powers = [power] * len(base_model_names)
                    best_pred = power_pred
                    
            except Exception as e:
                logger.warning(f"Failed to apply power {power}: {e}")
                continue
        
        logger.info(f"Power ensemble RMSE: {best_score:.4f}")
        
        return {
            "base_model_names": base_model_names,
            "best_powers": best_powers,
            "best_pred": best_pred,
            "best_score": best_score,
            "oof_predictions": model_predictions
        }
    
    def quantile_ensemble(
        self,
        base_model_names: List[str],
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        quantiles: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Create quantile-based ensemble.
        
        Args:
            base_model_names: List of base model names
            X: Feature matrix
            y: Target values
            cv_folds: Number of CV folds
            quantiles: List of quantiles to consider
            
        Returns:
            Dictionary containing the quantile ensemble
        """
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        logger.info(f"Creating quantile ensemble with {len(base_model_names)} base models")
        
        # Generate out-of-fold predictions
        stacking_features, model_predictions = self.cv.generate_stacking_features(
            base_model_names, X, y, cv_type="kfold", n_splits=cv_folds
        )
        
        # Calculate quantile-based predictions
        quantile_preds = []
        for q in quantiles:
            quantile_pred = np.quantile(stacking_features, q, axis=1)
            quantile_preds.append(quantile_pred)
        
        # Average the quantile predictions
        final_pred = np.mean(quantile_preds, axis=0)
        
        # Calculate score
        final_score = np.sqrt(mean_squared_error(y, final_pred))
        
        logger.info(f"Quantile ensemble RMSE: {final_score:.4f}")
        
        return {
            "base_model_names": base_model_names,
            "quantiles": quantiles,
            "quantile_preds": quantile_preds,
            "final_pred": final_pred,
            "final_score": final_score,
            "oof_predictions": model_predictions
        }
    
    def create_ensemble_comparison(
        self,
        base_model_names: List[str],
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple ensemble methods.
        
        Args:
            base_model_names: List of base model names
            X: Feature matrix
            y: Target values
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary comparing all ensemble methods
        """
        logger.info("Comparing ensemble methods")
        
        results = {}
        
        # Basic ensemble methods
        basic_results = self.cv.cv_ensemble_evaluation(
            base_model_names, X, y, cv_folds=cv_folds
        )
        results.update(basic_results)
        
        # Bayesian ensemble
        try:
            bayesian_result = self.bayesian_ensemble(base_model_names, X, y, cv_folds)
            results["bayesian"] = bayesian_result
        except Exception as e:
            logger.warning(f"Bayesian ensemble failed: {e}")
        
        # Rank ensemble
        try:
            rank_result = self.rank_ensemble(base_model_names, X, y, cv_folds)
            results["rank"] = rank_result
        except Exception as e:
            logger.warning(f"Rank ensemble failed: {e}")
        
        # Power ensemble
        try:
            power_result = self.power_ensemble(base_model_names, X, y, cv_folds)
            results["power"] = power_result
        except Exception as e:
            logger.warning(f"Power ensemble failed: {e}")
        
        # Quantile ensemble
        try:
            quantile_result = self.quantile_ensemble(base_model_names, X, y, cv_folds)
            results["quantile"] = quantile_result
        except Exception as e:
            logger.warning(f"Quantile ensemble failed: {e}")
        
        # Sort by score
        sorted_results = {}
        for method, result in results.items():
            score = result.get("final_score", result.get("oof_score", float('inf')))
            sorted_results[method] = {
                "result": result,
                "score": score
            }
        
        # Sort by score (ascending)
        sorted_results = dict(sorted(sorted_results.items(), key=lambda x: x[1]["score"]))
        
        logger.info("Ensemble comparison completed")
        for method, data in sorted_results.items():
            logger.info(f"{method}: {data['score']:.4f}")
        
        return sorted_results