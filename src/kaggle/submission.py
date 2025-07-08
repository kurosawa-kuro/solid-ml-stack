"""
Kaggle submission file generation utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import logging

from src.utils.config import Config
from src.ml.models.model_factory import ModelFactory
from src.ml.training.train import load_data


logger = logging.getLogger(__name__)


class KaggleSubmissionGenerator:
    """
    Generates Kaggle submission files from trained models.
    """
    
    def __init__(self, config: Config, output_dir: str = "submissions"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_single_model_submission(
        self,
        model_name: str,
        test_data: pd.DataFrame,
        id_column: str = "id",
        target_column: str = "target",
        filename: Optional[str] = None
    ) -> str:
        """
        Generate submission file for a single model.
        
        Args:
            model_name: Name of the model to use ('xgb', 'cat', 'lgbm')
            test_data: Test data for prediction
            id_column: Name of the ID column
            target_column: Name of the target column in submission
            filename: Optional custom filename
            
        Returns:
            Path to the generated submission file
        """
        # Load training data and train model
        train_data = load_data(self.config.db_path, "gold_house_features")
        
        # Prepare features (exclude ID and target columns)
        feature_columns = [col for col in train_data.columns if col not in [id_column, "price"]]
        X_train = train_data[feature_columns]
        y_train = train_data["price"]
        
        # Create and train model
        model = ModelFactory.create_model(model_name, self.config)
        model.fit(X_train, y_train)
        
        # Make predictions
        X_test = test_data[feature_columns]
        predictions = model.predict(X_test)
        
        # Create submission dataframe
        submission = pd.DataFrame({
            id_column: test_data[id_column],
            target_column: predictions
        })
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"submission_{model_name}_{timestamp}.csv"
        
        # Save submission file
        filepath = self.output_dir / filename
        submission.to_csv(filepath, index=False)
        
        logger.info(f"Generated submission file: {filepath}")
        return str(filepath)
    
    def generate_ensemble_submission(
        self,
        model_names: List[str],
        test_data: pd.DataFrame,
        ensemble_method: str = "average",
        weights: Optional[List[float]] = None,
        id_column: str = "id",
        target_column: str = "target",
        filename: Optional[str] = None
    ) -> str:
        """
        Generate submission file using ensemble of multiple models.
        
        Args:
            model_names: List of model names to ensemble
            test_data: Test data for prediction
            ensemble_method: Method for combining predictions ('average', 'weighted', 'median')
            weights: Weights for weighted average (if None, uses equal weights)
            id_column: Name of the ID column
            target_column: Name of the target column in submission
            filename: Optional custom filename
            
        Returns:
            Path to the generated submission file
        """
        # Load training data
        train_data = load_data(self.config.db_path, "gold_house_features")
        
        # Prepare features
        feature_columns = [col for col in train_data.columns if col not in [id_column, "price"]]
        X_train = train_data[feature_columns]
        y_train = train_data["price"]
        X_test = test_data[feature_columns]
        
        # Generate predictions from all models
        predictions = []
        for model_name in model_names:
            model = ModelFactory.create_model(model_name, self.config)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            predictions.append(pred)
        
        # Combine predictions based on ensemble method
        predictions_array = np.array(predictions)
        
        if ensemble_method == "average":
            final_predictions = np.mean(predictions_array, axis=0)
        elif ensemble_method == "weighted":
            if weights is None:
                weights = [1.0 / len(model_names)] * len(model_names)
            final_predictions = np.average(predictions_array, axis=0, weights=weights)
        elif ensemble_method == "median":
            final_predictions = np.median(predictions_array, axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")
        
        # Create submission dataframe
        submission = pd.DataFrame({
            id_column: test_data[id_column],
            target_column: final_predictions
        })
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            models_str = "_".join(model_names)
            filename = f"submission_ensemble_{models_str}_{ensemble_method}_{timestamp}.csv"
        
        # Save submission file
        filepath = self.output_dir / filename
        submission.to_csv(filepath, index=False)
        
        logger.info(f"Generated ensemble submission file: {filepath}")
        return str(filepath)
    
    def generate_stacking_submission(
        self,
        base_model_names: List[str],
        meta_model_name: str,
        test_data: pd.DataFrame,
        cv_folds: int = 5,
        id_column: str = "id",
        target_column: str = "target",
        filename: Optional[str] = None
    ) -> str:
        """
        Generate submission file using stacking ensemble.
        
        Args:
            base_model_names: List of base model names
            meta_model_name: Name of the meta-model
            test_data: Test data for prediction
            cv_folds: Number of cross-validation folds for stacking
            id_column: Name of the ID column
            target_column: Name of the target column in submission
            filename: Optional custom filename
            
        Returns:
            Path to the generated submission file
        """
        # Load training data
        train_data = load_data(self.config.db_path, "gold_house_features")
        
        # Prepare features
        feature_columns = [col for col in train_data.columns if col not in [id_column, "price"]]
        X_train = train_data[feature_columns]
        y_train = train_data["price"]
        X_test = test_data[feature_columns]
        
        # Create stacking model
        from src.ml.models.ensemble_model import StackingModel
        
        stacking_model = StackingModel(
            base_model_names=base_model_names,
            meta_model_name=meta_model_name,
            config=self.config
        )
        
        # Train stacking model
        stacking_model.fit(X_train, y_train)
        
        # Make predictions
        final_predictions = stacking_model.predict(X_test)
        
        # Create submission dataframe
        submission = pd.DataFrame({
            id_column: test_data[id_column],
            target_column: final_predictions
        })
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_models_str = "_".join(base_model_names)
            filename = f"submission_stacking_{base_models_str}_{meta_model_name}_{timestamp}.csv"
        
        # Save submission file
        filepath = self.output_dir / filename
        submission.to_csv(filepath, index=False)
        
        logger.info(f"Generated stacking submission file: {filepath}")
        return str(filepath)
    
    def generate_all_submissions(
        self,
        test_data: pd.DataFrame,
        id_column: str = "id",
        target_column: str = "target"
    ) -> Dict[str, str]:
        """
        Generate all types of submission files.
        
        Args:
            test_data: Test data for prediction
            id_column: Name of the ID column
            target_column: Name of the target column in submission
            
        Returns:
            Dictionary mapping submission type to file path
        """
        submissions = {}
        
        # Single model submissions
        for model_name in ["xgb", "cat", "lgbm"]:
            try:
                filepath = self.generate_single_model_submission(
                    model_name, test_data, id_column, target_column
                )
                submissions[f"single_{model_name}"] = filepath
            except Exception as e:
                logger.warning(f"Failed to generate {model_name} submission: {e}")
        
        # Ensemble submissions
        try:
            filepath = self.generate_ensemble_submission(
                ["xgb", "cat", "lgbm"], test_data, "average", None, id_column, target_column
            )
            submissions["ensemble_average"] = filepath
        except Exception as e:
            logger.warning(f"Failed to generate ensemble submission: {e}")
        
        # Stacking submission
        try:
            filepath = self.generate_stacking_submission(
                ["xgb", "cat", "lgbm"], "xgb", test_data, 5, id_column, target_column
            )
            submissions["stacking"] = filepath
        except Exception as e:
            logger.warning(f"Failed to generate stacking submission: {e}")
        
        return submissions