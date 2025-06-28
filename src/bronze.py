#!/usr/bin/env python3
"""Bronze layer - Import raw CSV data into DuckDB.

Reads raw CSV data and creates the bronze layer table in DuckDB.
This is the first step in the data processing pipeline.

Usage
-----
    python src/bronze.py [--db PATH] [--log-level LEVEL]

Defaults
--------
DB path : ./data/dwh/solid_ml.duckdb
Log level : INFO
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

from config import (
    Config, DatabaseManager, setup_logging, 
    log_step, log_success, log_error, 
    validate_dataframe, create_parser
)


def load_raw_data(config: Config, logger) -> pl.DataFrame:
    """Load raw CSV data from file."""
    log_step(logger, "Reading raw CSV data", 
             path=config.raw_data_path.relative_to(config.project_root))
    
    if not config.raw_data_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {config.raw_data_path}")
    
    df = pl.read_csv(config.raw_data_path)
    
    log_success(logger, "Raw data loaded", 
                rows=df.shape[0], columns=df.shape[1])
    
    return df


def create_bronze_table(df: pl.DataFrame, config: Config, logger) -> None:
    """Create bronze layer table in database."""
    log_step(logger, "Creating bronze layer table", 
             table=config.tables["raw"])
    
    db_manager = DatabaseManager(config)
    
    # Validate data before processing
    if not validate_dataframe(df, config.required_cols, logger):
        raise ValueError("Data validation failed")
    
    # Create table in database
    db_manager.replace_table(config.tables["raw"], df)
    
    # Verify table creation
    table_info = db_manager.get_table_info(config.tables["raw"])
    log_success(logger, "Bronze table created", 
                table=table_info["table_name"],
                rows=table_info["row_count"], 
                columns=table_info["column_count"])


def main(config: Config, logger) -> None:
    """Main bronze layer processing function."""
    try:
        # Load raw data
        df = load_raw_data(config, logger)
        
        # Create bronze table
        create_bronze_table(df, config, logger)
        
        log_success(logger, "Bronze layer processing completed successfully")
        
    except Exception as e:
        log_error(logger, "Bronze layer processing failed", error=e)
        sys.exit(1)


if __name__ == "__main__":
    # Parse arguments
    parser = create_parser("Create Bronze layer table in DuckDB.")
    args = parser.parse_args()
    
    # Setup configuration
    config = Config(db_path=args.db)
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Validate database path
    if not config.db_path.parent.exists():
        log_error(logger, f"Database directory does not exist: {config.db_path.parent}")
        sys.exit(1)
    
    # Run processing
    main(config, logger)
