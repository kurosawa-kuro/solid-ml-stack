#!/usr/bin/env python3
"""Silver layer - Data cleaning and feature derivation.

Reads bronze layer data and produces cleaned/derived silver layer table
according to the data preparation specification.

Usage
-----
    python src/silver.py [--db PATH] [--log-level LEVEL]

Defaults
--------
DB path : ./data/dwh/solid_ml.duckdb
Log level : INFO
"""
from __future__ import annotations

import sys
from functools import reduce
import operator

import polars as pl

from config import (
    Config, DatabaseManager, setup_logging, 
    log_step, log_success, log_error, 
    validate_dataframe, create_parser,
    clean_numeric_column, clean_string_column
)


def load_bronze_data(config: Config, logger) -> pl.DataFrame:
    """Load data from bronze layer table."""
    log_step(logger, "Loading bronze layer data", 
             table=config.tables["raw"])
    
    db_manager = DatabaseManager(config)
    
    if not db_manager.table_exists(config.tables["raw"]):
        raise ValueError(f"Bronze table '{config.tables['raw']}' not found. Run bronze.py first.")
    
    with db_manager as conn:
        df = conn.execute(f"SELECT * FROM {config.tables['raw']}").pl()
    
    log_success(logger, "Bronze data loaded", 
                rows=df.shape[0], columns=df.shape[1])
    
    return df


def clean_frame(df: pl.DataFrame, config: Config, logger) -> pl.DataFrame:
    """Apply silver layer cleaning and feature derivations."""
    log_step(logger, "Applying silver layer transformations")
    
    # Basic cleaning rules
    df = df.with_columns([
        clean_numeric_column(pl.col("price"), min_val=0).alias("price"),
        clean_numeric_column(pl.col("sqft"), min_val=0).alias("sqft"),
        clean_numeric_column(pl.col("bedrooms"), min_val=0).alias("bedrooms"),
        clean_numeric_column(pl.col("bathrooms"), min_val=0).alias("bathrooms"),
        clean_numeric_column(pl.col("year_built"), min_val=1900, max_val=config.current_year).alias("year_built"),
        clean_string_column(pl.col("location")).alias("location"),
        clean_string_column(pl.col("condition")).alias("condition"),
    ])
    
    log_step(logger, "Basic cleaning completed")
    
    # Derived columns
    df = df.with_columns([
        (pl.col("price") / pl.col("sqft")).alias("price_per_sqft"),
        (pl.lit(config.current_year) - pl.col("year_built")).alias("house_age"),
        (pl.col("bedrooms") / pl.col("bathrooms")).alias("bed_bath_ratio"),
    ])
    
    log_step(logger, "Derived features created")
    
    # Quality flags
    df = df.with_columns([
        ((pl.col("price_per_sqft") < 50) | (pl.col("price_per_sqft") > 1000)).alias("is_price_outlier"),
        ((pl.col("house_age") < 0) | (pl.col("house_age") > 100)).alias("is_age_outlier"),
        # Completeness: logical AND across required columns
        reduce(
            operator.and_,
            (pl.col(c).is_not_null() for c in config.required_cols),
        ).alias("is_complete_record"),
    ])
    
    log_step(logger, "Quality flags added")
    
    return df


def create_silver_table(df: pl.DataFrame, config: Config, logger) -> None:
    """Create silver layer table in database."""
    log_step(logger, "Creating silver layer table", 
             table=config.tables["silver"])
    
    db_manager = DatabaseManager(config)
    
    # Validate processed data
    if not validate_dataframe(df, config.required_cols, logger):
        raise ValueError("Silver layer data validation failed")
    
    # Create table in database
    db_manager.replace_table(config.tables["silver"], df)
    
    # Verify table creation
    table_info = db_manager.get_table_info(config.tables["silver"])
    log_success(logger, "Silver table created", 
                table=table_info["table_name"],
                rows=table_info["row_count"], 
                columns=table_info["column_count"])


def main(config: Config, logger) -> None:
    """Main silver layer processing function."""
    try:
        # Load bronze data
        df = load_bronze_data(config, logger)
        
        # Apply silver layer transformations
        silver_df = clean_frame(df, config, logger)
        
        # Create silver table
        create_silver_table(silver_df, config, logger)
        
        log_success(logger, "Silver layer processing completed successfully")
        
    except Exception as e:
        log_error(logger, "Silver layer processing failed", error=e)
        sys.exit(1)


if __name__ == "__main__":
    # Parse arguments
    parser = create_parser("Create Silver layer table in DuckDB.")
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