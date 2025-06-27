#!/usr/bin/env python3
"""Silver-layer transformation script.

Reads *raw_data* table from DuckDB and produces a cleaned/derived
*silver_house_data* table according to the data‑prep specification.

Usage
-----
    python src/silver.py [--db PATH]

Defaults
--------
DB path : ./data/dwh/solid_ml.duckdb
"""
from __future__ import annotations

import argparse
from datetime import datetime, UTC
from pathlib import Path
from functools import reduce
import operator

import duckdb
import polars as pl

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
CURRENT_YEAR = datetime.now(UTC).year

REQUIRED_COLS = [
    "price",
    "sqft",
    "bedrooms",
    "bathrooms",
    "year_built",
    "location",
    "condition",
]


def clean_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Apply Silver‑layer cleaning and feature derivations."""

    # Basic cleaning rules --------------------------------------------------
    df = (
        df.with_columns(
            [
                pl.when(pl.col("price") > 0)
                .then(pl.col("price"))
                .otherwise(None)
                .alias("price"),
                pl.when(pl.col("sqft") > 0)
                .then(pl.col("sqft"))
                .otherwise(None)
                .alias("sqft"),
                pl.when(pl.col("bedrooms") > 0)
                .then(pl.col("bedrooms"))
                .otherwise(None)
                .alias("bedrooms"),
                pl.when(pl.col("bathrooms") > 0)
                .then(pl.col("bathrooms"))
                .otherwise(None)
                .alias("bathrooms"),
                pl.when((pl.col("year_built") >= 1900) & (pl.col("year_built") <= CURRENT_YEAR))
                .then(pl.col("year_built"))
                .otherwise(None)
                .alias("year_built"),
                pl.col("location").str.strip_chars().str.to_uppercase().alias("location"),
                pl.col("condition").str.strip_chars().str.to_uppercase().alias("condition"),
            ]
        )
    )

    # Derived columns -------------------------------------------------------
    df = df.with_columns(
        [
            (pl.col("price") / pl.col("sqft")).alias("price_per_sqft"),
            (pl.lit(CURRENT_YEAR) - pl.col("year_built")).alias("house_age"),
            (pl.col("bedrooms") / pl.col("bathrooms")).alias("bed_bath_ratio"),
        ]
    )

    # Flags -----------------------------------------------------------------
    df = df.with_columns(
        [
            ((pl.col("price_per_sqft") < 50) | (pl.col("price_per_sqft") > 1000)).alias(
                "is_price_outlier"
            ),
            ((pl.col("house_age") < 0) | (pl.col("house_age") > 100)).alias(
                "is_age_outlier"
            ),
            # completeness: logical AND across required columns
            reduce(
                operator.and_,
                (pl.col(c).is_not_null() for c in REQUIRED_COLS),
            ).alias("is_complete_record"),
        ]
    )

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    try:
        raw_df = con.execute("SELECT * FROM raw_data").pl()
    except duckdb.CatalogException:
        raise SystemExit("raw_data table not found. Run import-raw-data.py first.")

    silver_df = clean_frame(raw_df)

    # Replace table
    con.execute("DROP TABLE IF EXISTS silver_house_data")
    con.register("silver_df", silver_df)
    con.execute("CREATE TABLE silver_house_data AS SELECT * FROM silver_df")

    print(
        f"✅ Silver table written with {silver_df.shape[0]} rows and {silver_df.shape[1]} columns."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Silver layer table in DuckDB.")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/dwh/solid_ml.duckdb"),
        help="Path to DuckDB database file.",
    )
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"Database not found: {args.db}")

    main(args.db)