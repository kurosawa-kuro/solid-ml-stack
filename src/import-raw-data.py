#!/usr/bin/env python3
"""Import raw CSV data into DuckDB.

Usage::
    python import-raw-data.py

This script reads ``data/raw/house_data.csv`` and inserts the records
into a DuckDB database file (``solid_ml.duckdb``) under the table
``raw_data``. If the table does not exist it will be created
automatically.
"""
from __future__ import annotations

import duckdb
from pathlib import Path
import polars as pl

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "house_data.csv"
DB_PATH = PROJECT_ROOT / "data" / "dwh" / "solid_ml.duckdb"
TABLE_NAME = "raw_data"

# ---------------------------------------------------------------------------
# Load CSV with Polars (fast & type‑aware)
# ---------------------------------------------------------------------------
print(f"📥 Reading CSV from {CSV_PATH.relative_to(PROJECT_ROOT)} …")

df = pl.read_csv(CSV_PATH)
print(f"   → {df.shape[0]:,} rows, {df.shape[1]} columns loaded")

# ---------------------------------------------------------------------------
# Write into DuckDB
# ---------------------------------------------------------------------------
print(f"🦆 Writing to DuckDB → {DB_PATH.relative_to(PROJECT_ROOT)}::{TABLE_NAME}")

con = duckdb.connect(DB_PATH)  # creates file if absent

# Register the Polars DataFrame as a DuckDB view
df_pandas = df.to_pandas()  # DuckDB 0.10+ can register Polars directly, but pandas is universally safe
con.register("raw_df", df_pandas)

# Create table (overwrite = REPLACE contents)
con.execute(f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} AS
SELECT * FROM raw_df
""")

# Or, if you prefer to truncate & reload each time:
# con.execute(f"DELETE FROM {TABLE_NAME}")
# con.execute(f"INSERT INTO {TABLE_NAME} SELECT * FROM raw_df")

con.close()
print("✅ Import completed.")
