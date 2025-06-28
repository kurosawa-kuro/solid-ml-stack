#!/usr/bin/env python3
"""Configuration and utilities for data processing pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, Optional
import logging
import sys

import duckdb
import polars as pl
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class Config:
    """Centralized configuration for the data processing pipeline."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.project_root = Path(__file__).resolve().parent.parent
        self.db_path = db_path or self.project_root / "data" / "dwh" / "solid_ml.duckdb"
        self.raw_data_path = self.project_root / "data" / "raw" / "house_data.csv"
        self.artifacts_dir = self.project_root / "target" / "preprocessing_artifacts"
        
        # Ensure directories exist
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Current year for calculations
        self.current_year = datetime.now(UTC).year
        
        # Required columns for data validation
        self.required_cols = [
            "price", "sqft", "bedrooms", "bathrooms", 
            "year_built", "location", "condition"
        ]
        
        # Table names
        self.tables = {
            "raw": "raw_data",
            "silver": "silver_house_data", 
            "gold": "gold_house_features",
            "ml": "ft_house_ml"
        }
        
        # Condition mapping
        self.condition_mapping = {
            'POOR': 1,
            'FAIR': 2, 
            'GOOD': 3,
            'EXCELLENT': 4
        }


# ---------------------------------------------------------------------------
# Database utilities
# ---------------------------------------------------------------------------
class DatabaseManager:
    """Database connection and operation manager."""
    
    def __init__(self, config: Config):
        self.config = config
        self.connection: Optional[duckdb.DuckDBPyConnection] = None
    
    def __enter__(self):
        self.connection = duckdb.connect(str(self.config.db_path))
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists in database."""
        with self as conn:
            try:
                conn.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
                return True
            except duckdb.CatalogException:
                return False
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get table information including row count and column count."""
        with self as conn:
            try:
                # Get row count
                row_result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                row_count = row_result[0] if row_result else 0
                
                # Get column count
                col_count = conn.execute(f"DESCRIBE {table_name}").df().shape[0]
                
                return {
                    "table_name": table_name,
                    "row_count": row_count,
                    "column_count": col_count,
                    "exists": True
                }
            except duckdb.CatalogException:
                return {
                    "table_name": table_name,
                    "row_count": 0,
                    "column_count": 0,
                    "exists": False
                }
    
    def replace_table(self, table_name: str, df: pl.DataFrame | pd.DataFrame) -> None:
        """Replace table with new data."""
        with self as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            if isinstance(df, pl.DataFrame):
                df_pandas = df.to_pandas()
            else:
                df_pandas = df
                
            conn.register("temp_df", df_pandas)
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM temp_df")


# ---------------------------------------------------------------------------
# Logging utilities
# ---------------------------------------------------------------------------
def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('data_processing.log')
        ]
    )
    return logging.getLogger(__name__)


def log_step(logger: logging.Logger, step_name: str, **kwargs):
    """Log processing step with optional data."""
    logger.info(f"🔄 {step_name}")
    for key, value in kwargs.items():
        if isinstance(value, (int, float)):
            logger.info(f"   → {key}: {value:,}")
        else:
            logger.info(f"   → {key}: {value}")


def log_success(logger: logging.Logger, message: str, **kwargs):
    """Log success message."""
    logger.info(f"✅ {message}")
    for key, value in kwargs.items():
        if isinstance(value, (int, float)):
            logger.info(f"   - {key}: {value:,}")
        else:
            logger.info(f"   - {key}: {value}")


def log_error(logger: logging.Logger, message: str, error: Optional[Exception] = None):
    """Log error message."""
    logger.error(f"❌ {message}")
    if error:
        logger.error(f"   Error: {str(error)}")


# ---------------------------------------------------------------------------
# Data validation utilities
# ---------------------------------------------------------------------------
def validate_dataframe(df: pl.DataFrame, required_cols: list[str], logger: logging.Logger) -> bool:
    """Validate dataframe has required columns and non-empty."""
    # Check required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        log_error(logger, f"Missing required columns: {missing_cols}")
        return False
    
    # Check if dataframe is empty
    if df.shape[0] == 0:
        log_error(logger, "Dataframe is empty")
        return False
    
    log_success(logger, "Data validation passed", 
                rows=df.shape[0], columns=df.shape[1])
    return True


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def create_parser(description: str) -> argparse.ArgumentParser:
    """Create standardized argument parser."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--db",
        type=Path,
        help="Path to DuckDB database file.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    return parser


# ---------------------------------------------------------------------------
# Artifact management
# ---------------------------------------------------------------------------
def save_artifacts(artifacts: Dict[str, Any], filename: str, config: Config) -> Path:
    """Save processing artifacts to file."""
    import pickle
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    artifacts_path = config.artifacts_dir / f"{filename}_{timestamp}.pkl"
    
    with open(artifacts_path, 'wb') as f:
        pickle.dump(artifacts, f)
    
    return artifacts_path


# ---------------------------------------------------------------------------
# Data cleaning utilities
# ---------------------------------------------------------------------------
def clean_numeric_column(col: pl.Expr, min_val: float = 0, max_val: Optional[float] = None) -> pl.Expr:
    """Clean numeric column with validation rules."""
    expr = pl.when(col > min_val).then(col).otherwise(None)
    if max_val is not None:
        expr = pl.when((col >= min_val) & (col <= max_val)).then(col).otherwise(None)
    return expr


def clean_string_column(col: pl.Expr) -> pl.Expr:
    """Clean string column by stripping and converting to uppercase."""
    return col.str.strip_chars().str.to_uppercase() 