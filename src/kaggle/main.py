"""
Main Kaggle competition utilities and workflows.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import logging

from src.utils.config import Config
from src.utils.base import setup_logging
from src.ml.training.train import load_data
from src.kaggle.submission import KaggleSubmissionGenerator
from src.kaggle.cross_validation import KaggleCrossValidator
from src.kaggle.ensemble import KaggleEnsemble


def kaggle_cv_workflow(
    config: Config,
    model_names: List[str],
    cv_type: str = "kfold",
    n_splits: int = 5,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Run cross-validation workflow for Kaggle competition.
    
    Args:
        config: Configuration object
        model_names: List of model names to evaluate
        cv_type: Type of cross-validation
        n_splits: Number of CV folds
        logger: Logger instance
    """
    if logger is None:
        logger = setup_logging("INFO")
    
    logger.info(f"Starting Kaggle CV workflow with {len(model_names)} models")
    
    # Load data
    data = load_data(config.db_path, "gold_house_features")
    
    # Prepare features and target
    feature_columns = [col for col in data.columns if col not in ["id", "price"]]
    X = data[feature_columns]
    y = data["price"]
    
    # Initialize cross-validator
    cv = KaggleCrossValidator(config)
    
    # Run cross-validation for multiple models
    results = cv.cross_validate_multiple_models(
        model_names, X, y, cv_type, n_splits
    )
    
    # Display results
    logger.info("=== Cross-Validation Results ===")
    for model_name, result in results.items():
        if "error" in result:
            logger.error(f"{model_name}: {result['error']}")
        else:
            score = result["oof_score"]
            logger.info(f"{model_name}: {score:.4f}")
    
    # Generate ensemble evaluation
    ensemble = KaggleEnsemble(config)
    ensemble_results = ensemble.create_ensemble_comparison(
        model_names, X, y, n_splits
    )
    
    logger.info("=== Ensemble Results ===")
    for method, data in ensemble_results.items():
        logger.info(f"{method}: {data['score']:.4f}")


def kaggle_submission_workflow(
    config: Config,
    test_data_path: str,
    model_names: Optional[List[str]] = None,
    ensemble_methods: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Generate submission files for Kaggle competition.
    
    Args:
        config: Configuration object
        test_data_path: Path to test data CSV
        model_names: List of model names to use
        ensemble_methods: List of ensemble methods to use
        logger: Logger instance
    """
    if logger is None:
        logger = setup_logging("INFO")
    
    if model_names is None:
        model_names = ["xgb", "cat", "lgbm"]
    
    if ensemble_methods is None:
        ensemble_methods = ["average", "bayesian", "stacking"]
    
    logger.info(f"Starting Kaggle submission workflow")
    
    # Load test data
    test_data = pd.read_csv(test_data_path)
    logger.info(f"Loaded test data with {len(test_data)} rows")
    
    # Initialize submission generator
    submission_gen = KaggleSubmissionGenerator(config)
    
    # Generate single model submissions
    for model_name in model_names:
        try:
            filepath = submission_gen.generate_single_model_submission(
                model_name, test_data
            )
            logger.info(f"Generated {model_name} submission: {filepath}")
        except Exception as e:
            logger.error(f"Failed to generate {model_name} submission: {e}")
    
    # Generate ensemble submissions
    if "average" in ensemble_methods:
        try:
            filepath = submission_gen.generate_ensemble_submission(
                model_names, test_data, "average"
            )
            logger.info(f"Generated average ensemble submission: {filepath}")
        except Exception as e:
            logger.error(f"Failed to generate average ensemble submission: {e}")
    
    if "stacking" in ensemble_methods:
        try:
            filepath = submission_gen.generate_stacking_submission(
                model_names, "xgb", test_data
            )
            logger.info(f"Generated stacking submission: {filepath}")
        except Exception as e:
            logger.error(f"Failed to generate stacking submission: {e}")
    
    # Generate all submissions
    try:
        all_submissions = submission_gen.generate_all_submissions(test_data)
        logger.info(f"Generated {len(all_submissions)} submission files")
        for submission_type, filepath in all_submissions.items():
            logger.info(f"  {submission_type}: {filepath}")
    except Exception as e:
        logger.error(f"Failed to generate all submissions: {e}")


def main():
    """Main entry point for Kaggle utilities."""
    parser = argparse.ArgumentParser(description="Kaggle competition utilities")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Cross-validation command
    cv_parser = subparsers.add_parser("cv", help="Run cross-validation")
    cv_parser.add_argument("--db", required=True, help="Database path")
    cv_parser.add_argument("--models", nargs="+", default=["xgb", "cat", "lgbm"],
                          help="Models to evaluate")
    cv_parser.add_argument("--cv-type", default="kfold", choices=["kfold", "stratified", "timeseries"],
                          help="Type of cross-validation")
    cv_parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    
    # Submission command
    sub_parser = subparsers.add_parser("submission", help="Generate submission files")
    sub_parser.add_argument("--db", required=True, help="Database path")
    sub_parser.add_argument("--test-data", required=True, help="Test data CSV path")
    sub_parser.add_argument("--models", nargs="+", default=["xgb", "cat", "lgbm"],
                           help="Models to use")
    sub_parser.add_argument("--ensemble-methods", nargs="+", default=["average", "stacking"],
                           help="Ensemble methods to use")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Setup logging
    logger = setup_logging("INFO")
    
    # Create config
    config = Config(db_path=args.db)
    
    if args.command == "cv":
        kaggle_cv_workflow(
            config, args.models, args.cv_type, args.n_splits, logger
        )
    elif args.command == "submission":
        kaggle_submission_workflow(
            config, args.test_data, args.models, args.ensemble_methods, logger
        )
    else:
        logger.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()