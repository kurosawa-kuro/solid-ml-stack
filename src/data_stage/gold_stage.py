#!/usr/bin/env python3
"""Gold layer - Feature engineering for ML.

Creates high-level features for machine learning from silver layer data.
This is the final step in the data processing pipeline before ML training.

Usage
-----
    python src/gold.py [--db PATH] [--log-level LEVEL]

Defaults
--------
DB path : ./data/dwh/solid_ml.duckdb
Log level : INFO
"""
from __future__ import annotations

import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# srcディレクトリをPYTHONPATHに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.config import (
    Config, DatabaseManager, setup_logging, 
    log_step, log_success, log_error, 
    validate_dataframe, create_parser, save_artifacts
)


def load_silver_data(config: Config, logger) -> pd.DataFrame:
    """Load data from silver layer table (complete records only)."""
    log_step(logger, "Loading silver layer data", 
             table=config.tables["silver"])
    
    db_manager = DatabaseManager(config)
    
    if not db_manager.table_exists(config.tables["silver"]):
        raise ValueError(f"Silver table '{config.tables['silver']}' not found. Run silver.py first.")
    
    with db_manager as conn:
        query = f"""
        SELECT * FROM {config.tables['silver']} 
        WHERE is_complete_record = true
        """
        df = conn.execute(query).df()
    
    if df.empty:
        raise ValueError("No complete records found in silver layer")
    
    log_success(logger, "Silver data loaded", 
                rows=df.shape[0], columns=df.shape[1])
    
    return df


def create_basic_features(df: pd.DataFrame, config: Config, logger) -> pd.DataFrame:
    """Create basic derived features."""
    log_step(logger, "Creating basic derived features")
    
    # 1. Logarithmic transformations
    df['log_price'] = np.log1p(df['price'])
    df['log_sqft'] = np.log1p(df['sqft'])
    
    # 2. Polynomial features
    df['sqft_squared'] = df['sqft'] ** 2
    df['price_per_sqft_squared'] = df['price_per_sqft'] ** 2
    df['sqft_cubed'] = df['sqft'] ** 3
    
    # 3. Interaction features
    df['price_bedrooms_interaction'] = df['price'] * df['bedrooms']
    df['price_bathrooms_interaction'] = df['price'] * df['bathrooms']
    df['sqft_bedrooms_interaction'] = df['sqft'] * df['bedrooms']
    df['sqft_bathrooms_interaction'] = df['sqft'] * df['bathrooms']
    df['price_sqft_ratio'] = df['price'] / df['sqft']
    df['bedrooms_bathrooms_interaction'] = df['bedrooms'] * df['bathrooms']
    
    log_step(logger, "Basic features created")
    return df


def create_categorical_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Create categorical features based on domain knowledge."""
    log_step(logger, "Creating categorical features")
    
    # Age-based categories
    df['is_new_house'] = (df['house_age'] <= 10).astype(int)
    df['is_medium_age'] = ((df['house_age'] > 10) & (df['house_age'] <= 50)).astype(int)
    df['is_old_house'] = (df['house_age'] > 50).astype(int)
    
    # Size-based categories (quartiles)
    sqft_q25 = df['sqft'].quantile(0.25)
    sqft_q75 = df['sqft'].quantile(0.75)
    df['is_small_house'] = (df['sqft'] <= sqft_q25).astype(int)
    df['is_large_house'] = (df['sqft'] >= sqft_q75).astype(int)
    
    # Price-based categories (quartiles)
    price_q25 = df['price'].quantile(0.25)
    price_q75 = df['price'].quantile(0.75)
    df['is_affordable'] = (df['price'] <= price_q25).astype(int)
    df['is_expensive'] = (df['price'] >= price_q75).astype(int)
    
    log_step(logger, "Categorical features created")
    return df


def create_location_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Create location-based features."""
    log_step(logger, "Creating location-based features")
    
    # Location average prices
    location_avg = df.groupby('location')['price'].mean().reset_index()
    location_avg.columns = ['location', 'location_avg_price']
    df = df.merge(location_avg, on='location', how='left')
    
    # Price vs location average
    df['price_vs_location_avg'] = df['price'] - df['location_avg_price']
    df['price_vs_location_avg_ratio'] = df['price'] / df['location_avg_price']
    
    # Location price rank
    df['location_price_rank'] = df.groupby('location')['price'].rank(pct=True)
    
    log_step(logger, "Location features created")
    return df


def create_condition_features(df: pd.DataFrame, config: Config, logger) -> pd.DataFrame:
    """Create condition-based features."""
    log_step(logger, "Creating condition-based features")
    
    # Condition score - fix FutureWarning
    df['condition_score'] = df['condition'].replace(config.condition_mapping).astype(int)
    
    # Additional derived features
    df['price_efficiency'] = df['sqft'] / df['price']
    df['room_density'] = (df['bedrooms'] + df['bathrooms']) / df['sqft']
    df['age_price_interaction'] = df['house_age'] * df['price']
    
    # Quality flags
    df['is_premium_location'] = df['location'].isin(['WATERFRONT', 'DOWNTOWN']).astype(int)
    df['is_high_condition'] = (df['condition_score'] >= 3).astype(int)
    
    log_step(logger, "Condition features created")
    return df


def create_composite_scores(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Create composite quality and value scores."""
    log_step(logger, "Creating composite scores")
    
    # Overall quality score
    df['overall_quality_score'] = (
        df['condition_score'] * 0.4 + 
        df['is_premium_location'] * 0.3 + 
        df['is_new_house'] * 0.3
    )
    
    # Price reasonableness score
    df['price_reasonableness_score'] = (
        (df['price_vs_location_avg_ratio'] - 1).abs() * -1 + 1
    )
    
    log_step(logger, "Composite scores created")
    return df


def clean_features(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Clean features by handling infinities and NaN values."""
    log_step(logger, "Cleaning features")
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col].median())
    
    log_step(logger, "Features cleaned")
    return df


def create_gold_features(df: pd.DataFrame, config: Config, logger) -> pd.DataFrame:
    """Create all gold layer features."""
    log_step(logger, "Starting gold layer feature engineering")
    
    # Apply feature engineering steps
    df = create_basic_features(df, config, logger)
    df = create_categorical_features(df, logger)
    df = create_location_features(df, logger)
    df = create_condition_features(df, config, logger)
    df = create_composite_scores(df, logger)
    df = clean_features(df, logger)
    
    log_success(logger, "Gold layer features created", 
                features=len(df.columns))
    
    return df


def create_analytics_view(db_manager: DatabaseManager, config: Config, logger) -> None:
    """Create analytics view for gold layer data."""
    log_step(logger, "Creating analytics view")
    
    with db_manager as conn:
        conn.execute("DROP VIEW IF EXISTS v_house_analytics")
        
        analytics_view_query = """
        CREATE VIEW v_house_analytics AS
        SELECT 
            *,
            CASE 
                WHEN overall_quality_score >= 3.5 THEN 'Premium'
                WHEN overall_quality_score >= 2.5 THEN 'Standard'
                ELSE 'Basic'
            END as quality_tier,
            CASE 
                WHEN price_reasonableness_score >= 0.8 THEN 'Good Value'
                WHEN price_reasonableness_score >= 0.6 THEN 'Fair Value'
                ELSE 'Overpriced'
            END as value_assessment
        FROM gold_house_features
        """
        
        conn.execute(analytics_view_query)
    
    log_success(logger, "Analytics view created")


def save_gold_artifacts(df: pd.DataFrame, config: Config, logger) -> Path:
    """Save gold layer artifacts."""
    log_step(logger, "Saving gold layer artifacts")
    
    feature_names = list(df.columns)
    feature_names.remove('price')  # Remove target variable
    
    artifacts = {
        'feature_names': feature_names,
        'feature_stats': {
            'feature_count': len(df.columns),
            'record_count': len(df),
            'numeric_features': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': list(df.select_dtypes(include=['object']).columns),
            'processing_timestamp': datetime.now().isoformat()
        },
        'condition_mapping': config.condition_mapping,
        'processing_metadata': {
            'source_table': config.tables["silver"],
            'target_table': config.tables["gold"],
            'total_features': len(feature_names),
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    artifacts_path = save_artifacts(artifacts, "gold_features", config)
    log_success(logger, "Gold artifacts saved", path=artifacts_path)
    
    return artifacts_path


def create_gold_table(df: pd.DataFrame, config: Config, logger) -> None:
    """Create gold layer table in database."""
    log_step(logger, "Creating gold layer table", 
             table=config.tables["gold"])
    
    db_manager = DatabaseManager(config)
    
    # Create table
    db_manager.replace_table(config.tables["gold"], df)
    
    # Create analytics view
    create_analytics_view(db_manager, config, logger)
    
    # Verify table creation
    table_info = db_manager.get_table_info(config.tables["gold"])
    log_success(logger, "Gold table created", 
                table=table_info["table_name"],
                rows=table_info["row_count"], 
                columns=table_info["column_count"])


def main(config: Config, logger) -> None:
    """Main gold layer processing function."""
    try:
        # Load silver data
        df = load_silver_data(config, logger)
        
        # Create gold features
        gold_df = create_gold_features(df, config, logger)
        
        # Create gold table
        create_gold_table(gold_df, config, logger)
        
        # Save artifacts
        save_gold_artifacts(gold_df, config, logger)
        
        log_success(logger, "Gold layer processing completed successfully")
        
    except Exception as e:
        log_error(logger, "Gold layer processing failed", error=e)
        sys.exit(1)


if __name__ == "__main__":
    # Parse arguments
    parser = create_parser("Create Gold layer features in DuckDB.")
    args = parser.parse_args()
    
    # Setup configuration
    config = Config(db_path=args.db)
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Validate database exists
    if not config.db_path.exists():
        log_error(logger, f"Database not found: {config.db_path}")
        sys.exit(1)
    
    # Run processing
    main(config, logger)
