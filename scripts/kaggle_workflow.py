#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('src')

from preprocessing import Preprocessor
from features.engineering import FeatureEngineeringPipeline
from modeling import ModelFactory
from optimization import OptimizerFactory
# from evaluation import ModelEvaluator  # TODO: Not implemented yet
from submission import SubmissionGenerator


def main():
    parser = argparse.ArgumentParser(description='Kaggle ML Workflow')
    parser.add_argument('--train-path', required=True, help='Path to training data')
    parser.add_argument('--test-path', required=True, help='Path to test data')
    parser.add_argument('--target-col', required=True, help='Target column name')
    parser.add_argument('--id-col', default='id', help='ID column name')
    parser.add_argument('--problem-type', choices=['regression', 'classification'], 
                       default='regression', help='Problem type')
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    parser.add_argument('--optimize', action='store_true', help='Run hyperparameter optimization')
    parser.add_argument('--ensemble', action='store_true', help='Create ensemble models')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=== Kaggle ML Workflow ===")
    print(f"Training data: {args.train_path}")
    print(f"Test data: {args.test_path}")
    print(f"Target column: {args.target_col}")
    print(f"Problem type: {args.problem_type}")
    print(f"Output directory: {args.output_dir}")
    
    print("\n1. Loading data...")
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    print("\n2. Preprocessing data...")
    preprocessor = Preprocessor()
    
    X_train, X_val, y_train, y_val = preprocessor.prepare_data(
        train_df, args.target_col, test_size=0.2, random_state=42
    )
    
    X_train_processed, X_val_processed = preprocessor.process_train_test(
        X_train, X_val, y_train
    )
    
    print(f"Processed train shape: {X_train_processed.shape}")
    print(f"Processed validation shape: {X_val_processed.shape}")
    
    print("\n3. Feature engineering...")
    feature_engineer = FeatureEngineeringPipeline()
    feature_engineer.add_numeric_features()
    feature_engineer.add_categorical_features()
    feature_engineer.add_interaction_features()
    feature_engineer.add_datetime_features()
    
    X_train_features = feature_engineer.fit_transform(X_train_processed, y_train)
    X_val_features = feature_engineer.transform(X_val_processed)
    
    print(f"Features shape: {X_train_features.shape}")
    
    print("\n4. Creating models...")
    factory = ModelFactory()
    models = factory.get_default_models(target_type=args.problem_type)
    
    if args.ensemble:
        ensemble_models = factory.create_ensemble_models(target_type=args.problem_type)
        models.extend(ensemble_models)
    
    print(f"Created {len(models)} models")
    
    print("\n5. Training models...")
    trained_models = {}
    
    for model in models:
        model_name = getattr(model.config, 'name', str(type(model).__name__))
        print(f"\nTraining {model_name}...")
        
        # Simple training without optimization for now
        model.fit(X_train_features, y_train, X_val_features, y_val)
        trained_models[model_name] = model
        
        # Basic validation score
        val_pred = model.predict(X_val_features)
        from sklearn.metrics import mean_squared_error
        val_score = mean_squared_error(y_val, val_pred) ** 0.5
        print(f"  Validation RMSE: {val_score:.4f}")
    
    print("\n6. Model training completed...")
    
    print("\n7. Generating predictions...")
    # Process test data with same structure as training data (keep all columns except target)
    # Note: test data doesn't have target column, so use all columns
    X_test_processed = preprocessor.transform(test_df.drop(columns=[args.target_col] if args.target_col in test_df.columns else []))
    X_test_features = feature_engineer.transform(X_test_processed)
    
    predictions = {}
    for model_name, model in trained_models.items():
        pred = model.predict(X_test_features)
        predictions[model_name] = pred
    
    print("\n8. Creating submissions...")
    submission_generator = SubmissionGenerator(str(output_dir / 'submissions'))
    
    test_ids = test_df[args.id_col].values
    
    submission_paths = submission_generator.create_multiple_submissions(
        predictions, test_ids, args.target_col, args.id_col
    )
    
    ensemble_path = submission_generator.create_ensemble_submission(
        predictions, test_ids=test_ids, target_column=args.target_col, 
        id_column=args.id_col, filename='ensemble_submission.csv'
    )
    
    print(f"\nCreated {len(submission_paths) + 1} submissions:")
    for path in submission_paths + [ensemble_path]:
        print(f"  {path}")
    
    print(f"\nWorkflow completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()