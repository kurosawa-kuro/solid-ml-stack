"""
Example Kaggle workflow demonstrating the refactored features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

from src.utils.config import Config
from src.utils.base import setup_logging
from src.ml.training.train import load_data
from src.kaggle.submission import KaggleSubmissionGenerator
from src.kaggle.cross_validation import KaggleCrossValidator
from src.kaggle.ensemble import KaggleEnsemble


def main():
    """
    Example Kaggle workflow showcasing bronze medal strategies.
    """
    # Setup logging
    logger = setup_logging("INFO")
    logger.info("Starting Kaggle workflow example")
    
    # Configuration
    config = Config(db_path="data/dwh/solid_ml.duckdb")
    
    # Load data
    try:
        data = load_data(config.db_path, "gold_house_features")
        logger.info(f"Loaded data with {len(data)} rows and {len(data.columns)} columns")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Prepare features and target
    feature_columns = [col for col in data.columns if col not in ["id", "price"]]
    X = data[feature_columns]
    y = data["price"]
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    
    # 1. Cross-validation evaluation
    logger.info("=== Step 1: Cross-validation evaluation ===")
    
    cv = KaggleCrossValidator(config)
    model_names = ["xgb", "cat", "lgbm"]
    
    # Single model CV
    cv_results = cv.cross_validate_multiple_models(
        model_names, X, y, cv_type="kfold", n_splits=5
    )
    
    logger.info("Cross-validation results:")
    for model_name, result in cv_results.items():
        if "error" in result:
            logger.error(f"{model_name}: {result['error']}")
        else:
            score = result["oof_score"]
            std = result["aggregated_metrics"]["rmse_std"]
            logger.info(f"{model_name}: {score:.4f} (+/- {std:.4f})")
    
    # 2. Ensemble evaluation
    logger.info("=== Step 2: Ensemble evaluation ===")
    
    ensemble = KaggleEnsemble(config)
    ensemble_results = ensemble.create_ensemble_comparison(
        model_names, X, y, cv_folds=5
    )
    
    logger.info("Ensemble comparison results:")
    for method, data in ensemble_results.items():
        logger.info(f"{method}: {data['score']:.4f}")
    
    # 3. Submission file generation (mock example)
    logger.info("=== Step 3: Submission file generation ===")
    
    # Create mock test data
    test_data = X.sample(n=100, random_state=42).copy()
    test_data["id"] = range(1, len(test_data) + 1)
    
    # Initialize submission generator
    submission_gen = KaggleSubmissionGenerator(config, output_dir="submissions")
    
    # Generate single model submissions
    for model_name in model_names:
        try:
            filepath = submission_gen.generate_single_model_submission(
                model_name, test_data, id_column="id", target_column="target"
            )
            logger.info(f"Generated {model_name} submission: {filepath}")
        except Exception as e:
            logger.error(f"Failed to generate {model_name} submission: {e}")
    
    # Generate ensemble submission
    try:
        filepath = submission_gen.generate_ensemble_submission(
            model_names, test_data, ensemble_method="average",
            id_column="id", target_column="target"
        )
        logger.info(f"Generated ensemble submission: {filepath}")
    except Exception as e:
        logger.error(f"Failed to generate ensemble submission: {e}")
    
    # Generate all submissions
    try:
        all_submissions = submission_gen.generate_all_submissions(
            test_data, id_column="id", target_column="target"
        )
        logger.info(f"Generated {len(all_submissions)} submission files:")
        for submission_type, filepath in all_submissions.items():
            logger.info(f"  {submission_type}: {filepath}")
    except Exception as e:
        logger.error(f"Failed to generate all submissions: {e}")
    
    # 4. Advanced ensemble techniques
    logger.info("=== Step 4: Advanced ensemble techniques ===")
    
    # Bayesian ensemble
    try:
        bayesian_result = ensemble.bayesian_ensemble(model_names, X, y, cv_folds=3)
        logger.info(f"Bayesian ensemble score: {bayesian_result['final_score']:.4f}")
        logger.info("Optimal weights:")
        for model, weight in bayesian_result['weight_mapping'].items():
            logger.info(f"  {model}: {weight:.4f}")
    except Exception as e:
        logger.error(f"Bayesian ensemble failed: {e}")
    
    # Rank ensemble
    try:
        rank_result = ensemble.rank_ensemble(model_names, X, y, cv_folds=3)
        logger.info(f"Rank ensemble score: {rank_result['final_score']:.4f}")
    except Exception as e:
        logger.error(f"Rank ensemble failed: {e}")
    
    logger.info("Kaggle workflow example completed successfully!")


if __name__ == "__main__":
    main()