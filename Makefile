# Makefile for solid-ml-stack

.PHONY: bronze silver gold pipeline status clean help ml-xgb ml-cat ml-lgbm ml-ensemble ml-stack ml-all ml-help kaggle-cv kaggle-submission kaggle-help

# Individual layer processing
bronze:
	python3 src/pipelines/main.py --steps bronze

silver:
	python3 src/pipelines/main.py --steps silver

gold:
	python3 src/pipelines/main.py --steps gold

# Complete pipeline execution
pipeline:
	python3 src/pipelines/main.py

# Pipeline with specific steps
bronze-silver:
	python3 src/pipelines/main.py --steps bronze silver

silver-gold:
	python3 src/pipelines/main.py --steps silver gold

# Check pipeline status
status:
	python3 src/pipelines/main.py --status-only

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
	@echo "  ml-help       - Show ML-specific help"
	@echo "  kaggle-help   - Show Kaggle competition help"
	@echo "  help          - Show this help message"

# Quick test targets
test-bronze:
	python3 src/bronze.py --log-level INFO

test-silver:
	python3 src/silver.py --log-level INFO

test-gold:
	python3 src/gold.py --log-level INFO

test-pipeline:
	python3 src/pipelines/main.py --log-level INFO

# MLモデル実行
ML_DB ?= data/dwh/solid_ml.duckdb

ml-xgb:
	python3 -m src.ml.training.train --db $(ML_DB) --model xgb

ml-xgb-cv:
	python3 -m src.ml.training.train --db $(ML_DB) --model xgb --cv-folds 5

ml-cat:
	python3 -m src.ml.training.train --db $(ML_DB) --model cat

ml-lgbm:
	python3 -m src.ml.training.train --db $(ML_DB) --model lgbm

ml-ensemble:
	python3 -m src.ml.training.train --db $(ML_DB) --model ensemble

ml-stack:
	python3 -m src.ml.training.train --db $(ML_DB) --model stacking

ml-all:
	$(MAKE) ml-xgb
	$(MAKE) ml-cat
	$(MAKE) ml-lgbm
	$(MAKE) ml-ensemble
	$(MAKE) ml-stack
	python3 -m src.ml.training.ml_report

ml-report:
	python3 -m src.ml.training.ml_report

ml-help:
	@echo "ML targets:"
	@echo "  ml-xgb       - XGBoost モデルの学習・推論"
	@echo "  ml-xgb-cv    - XGBoost モデルのクロスバリデーション"
	@echo "  ml-cat       - CatBoost モデルの学習・推論"
	@echo "  ml-lgbm      - LightGBM モデルの学習・推論"
	@echo "  ml-ensemble  - アンサンブル推論"
	@echo "  ml-stack     - スタッキング推論"
	@echo "  ml-all       - 全MLモデル一括実行"
	@echo "  ml-report    - ML実験結果レポート生成"

# Kaggle competition targets
kaggle-cv:
	python3 -m src.kaggle.main cv --db $(ML_DB) --models xgb cat lgbm --cv-type kfold --n-splits 5

kaggle-cv-all:
	python3 -m src.kaggle.main cv --db $(ML_DB) --models xgb cat lgbm --cv-type kfold --n-splits 5

kaggle-submission:
	python3 -m src.kaggle.main submission --db $(ML_DB) --test-data data/test.csv --models xgb cat lgbm

kaggle-submission-all:
	python3 -m src.kaggle.main submission --db $(ML_DB) --test-data data/test.csv --models xgb cat lgbm --ensemble-methods average bayesian stacking

kaggle-help:
	@echo "Kaggle competition targets:"
	@echo "  kaggle-cv           - Run cross-validation for Kaggle models"
	@echo "  kaggle-cv-all       - Run comprehensive CV evaluation"
	@echo "  kaggle-submission   - Generate basic submission files"
	@echo "  kaggle-submission-all - Generate all submission files with ensembles"
	@echo "  kaggle-help         - Show Kaggle-specific help"
