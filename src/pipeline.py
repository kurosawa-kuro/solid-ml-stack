#!/usr/bin/env python3
"""Complete data processing pipeline.

Executes the entire data processing pipeline from bronze to gold layers.
This script orchestrates the complete data flow.

Usage
-----
    python src/pipeline.py [--db PATH] [--log-level LEVEL] [--steps STEPS]

Defaults
--------
DB path : ./data/dwh/solid_ml.duckdb
Log level : INFO
Steps : bronze,silver,gold
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from config import (
    Config, DatabaseManager, setup_logging, 
    log_step, log_success, log_error, create_parser
)

# Import processing modules
from bronze import main as bronze_main
from silver import main as silver_main
from gold import main as gold_main


def validate_pipeline_dependencies(config: Config, steps: List[str], logger) -> bool:
    """Validate dependencies for pipeline steps."""
    log_step(logger, "Validating pipeline dependencies")
    
    db_manager = DatabaseManager(config)
    
    # Check required files and tables
    dependencies = {
        "bronze": {
            "files": [config.raw_data_path],
            "tables": []
        },
        "silver": {
            "files": [],
            "tables": [config.tables["raw"]]
        },
        "gold": {
            "files": [],
            "tables": [config.tables["silver"]]
        }
    }
    
    for step in steps:
        if step not in dependencies:
            log_error(logger, f"Unknown pipeline step: {step}")
            return False
        
        # Check files
        for file_path in dependencies[step]["files"]:
            if not file_path.exists():
                log_error(logger, f"Required file not found: {file_path}")
                return False
        
        # Check tables
        for table_name in dependencies[step]["tables"]:
            if not db_manager.table_exists(table_name):
                log_error(logger, f"Required table not found: {table_name}")
                return False
    
    log_success(logger, "Pipeline dependencies validated")
    return True


def execute_pipeline_step(step: str, config: Config, logger) -> bool:
    """Execute a single pipeline step."""
    log_step(logger, f"Executing {step} layer")
    
    try:
        if step == "bronze":
            bronze_main(config, logger)
        elif step == "silver":
            silver_main(config, logger)
        elif step == "gold":
            gold_main(config, logger)
        else:
            log_error(logger, f"Unknown pipeline step: {step}")
            return False
        
        log_success(logger, f"{step} layer completed")
        return True
        
    except Exception as e:
        log_error(logger, f"{step} layer failed", error=e)
        return False


def get_pipeline_status(config: Config, logger) -> dict:
    """Get status of all pipeline tables."""
    log_step(logger, "Checking pipeline status")
    
    db_manager = DatabaseManager(config)
    status = {}
    
    for layer, table_name in config.tables.items():
        table_info = db_manager.get_table_info(table_name)
        status[layer] = table_info
    
    # Log status
    for layer, info in status.items():
        if info["exists"]:
            log_success(logger, f"{layer} layer ready", 
                       rows=info["row_count"], 
                       columns=info["column_count"])
        else:
            log_error(logger, f"{layer} layer missing")
    
    return status


def print_pipeline_summary(status: dict, logger) -> None:
    """Print pipeline execution summary."""
    log_step(logger, "Pipeline Summary")
    
    total_rows = 0
    total_columns = 0
    
    for layer, info in status.items():
        if info["exists"]:
            rows = info["row_count"]
            cols = info["column_count"]
            total_rows = max(total_rows, rows)  # All should have same row count
            total_columns += cols
            
            logger.info(f"  {layer:8} | {rows:4} rows | {cols:3} columns")
    
    logger.info(f"  {'Total':8} | {total_rows:4} rows | {total_columns:3} columns")
    
    # Feature engineering summary
    if status.get("gold", {}).get("exists"):
        gold_cols = status["gold"]["column_count"]
        raw_cols = status.get("raw", {}).get("column_count", 0)
        engineered_features = gold_cols - raw_cols
        
        logger.info(f"  {'Features':8} | {raw_cols:4} original | +{engineered_features:3} engineered")


def main(config: Config, steps: List[str], logger) -> None:
    """Main pipeline execution function."""
    log_step(logger, "Starting data processing pipeline", steps=", ".join(steps))
    
    # Validate dependencies
    if not validate_pipeline_dependencies(config, steps, logger):
        sys.exit(1)
    
    # Execute pipeline steps
    for step in steps:
        if not execute_pipeline_step(step, config, logger):
            log_error(logger, f"Pipeline failed at {step} step")
            sys.exit(1)
    
    # Final status check
    status = get_pipeline_status(config, logger)
    
    # Print summary
    print_pipeline_summary(status, logger)
    
    log_success(logger, "Data processing pipeline completed successfully")


if __name__ == "__main__":
    # Parse arguments
    parser = create_parser("Execute complete data processing pipeline.")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["bronze", "silver", "gold"],
        default=["bronze", "silver", "gold"],
        help="Pipeline steps to execute (default: all steps)"
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Only show pipeline status without execution"
    )
    args = parser.parse_args()
    
    # Setup configuration
    config = Config(db_path=args.db)
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Validate database path
    if not config.db_path.parent.exists():
        log_error(logger, f"Database directory does not exist: {config.db_path.parent}")
        sys.exit(1)
    
    if args.status_only:
        # Show status only
        get_pipeline_status(config, logger)
    else:
        # Execute pipeline
        main(config, args.steps, logger) 