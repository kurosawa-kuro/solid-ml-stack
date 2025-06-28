# Makefile for solid-ml-stack

.PHONY: bronze silver gold pipeline status clean help ml-xgb ml-cat ml-lgbm ml-ensemble ml-stack ml-all ml-help

# Individual layer processing
bronze:
	python3 src/pipeline.py --steps bronze

silver:
	python3 src/pipeline.py --steps silver

gold:
	python3 src/pipeline.py --steps gold

# Complete pipeline execution
pipeline:
	python3 src/pipeline.py

# Pipeline with specific steps
bronze-silver:
	python3 src/pipeline.py --steps bronze silver

silver-gold:
	python3 src/pipeline.py --steps silver gold

# Check pipeline status
status:
	python3 src/pipeline.py --status-only

# Clean generated files
clean:
	rm -rf data/dwh/solid_ml.duckdb
	rm -rf target/preprocessing_artifacts/*
	rm -f data_processing.log

# Development helpers
dev-setup:
	python3 -m pip install --break-system-packages -r requirements.txt
	mkdir -p data/raw data/dwh target/preprocessing_artifacts

# Run all data processing stages (legacy alias)
run-all: pipeline

# Default target
run: pipeline

# Help
help:
	@echo "Available targets:"
	@echo "  bronze        - Process bronze layer only"
	@echo "  silver        - Process silver layer only"
	@echo "  gold          - Process gold layer only"
	@echo "  pipeline      - Run complete pipeline (bronze -> silver -> gold)"
	@echo "  bronze-silver - Run bronze and silver layers"
	@echo "  silver-gold   - Run silver and gold layers"
	@echo "  status        - Check pipeline status"
	@echo "  clean         - Remove generated files"
	@echo "  dev-setup     - Setup development environment"
	@echo "  test-*        - Quick test individual layers"
	@echo "  help          - Show this help message"

# Quick test targets
test-bronze:
	python3 src/bronze.py --log-level INFO

test-silver:
	python3 src/silver.py --log-level INFO

test-gold:
	python3 src/gold.py --log-level INFO

test-pipeline:
	python3 src/pipeline.py --log-level INFO

# MLモデル実行
ML_DB ?= data/dwh/solid_ml.duckdb

ml-xgb:
	python3 src/pipelines/train.py --db $(ML_DB) --model xgb

ml-xgb-cv:
	python3 src/pipelines/train.py --db $(ML_DB) --model xgb --cv-folds 5

ml-cat:
	python3 src/pipelines/train.py --db $(ML_DB) --model cat

ml-lgbm:
	python3 src/pipelines/train.py --db $(ML_DB) --model lgbm

ml-ensemble:
	python3 src/pipelines/train.py --db $(ML_DB) --model ensemble

ml-stack:
	python3 src/pipelines/train.py --db $(ML_DB) --model stacking

ml-all:
	$(MAKE) ml-xgb
	$(MAKE) ml-cat
	$(MAKE) ml-lgbm
	$(MAKE) ml-ensemble
	$(MAKE) ml-stack
	#	python3 src/ml/ml_report.py

ml-help:
	@echo "ML targets:"
	@echo "  ml-xgb       - XGBoost モデルの学習・推論"
	@echo "  ml-xgb-cv    - XGBoost モデルのクロスバリデーション"
	@echo "  ml-cat       - CatBoost モデルの学習・推論"
	@echo "  ml-lgbm      - LightGBM モデルの学習・推論"
	@echo "  ml-ensemble  - アンサンブル推論"
	@echo "  ml-stack     - スタッキング推論"
	@echo "  ml-all       - 全MLモデル一括実行"
