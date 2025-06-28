# Makefile for solid-ml-stack

.PHONY: bronze silver gold pipeline status clean help

# Individual layer processing
bronze:
	python3 src/bronze.py

silver:
	python3 src/silver.py

gold:
	python3 src/gold.py

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
