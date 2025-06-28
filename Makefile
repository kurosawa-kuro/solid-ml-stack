# Makefile for solid-ml-stack

.PHONY: import-raw-data bronze silver gold run-all
import-raw-data:
	python3 src/blonze.py

bronze:
	python3 src/blonze.py

silver:
	python3 src/silver.py

gold:
	python3 src/gold.py

# Run all data processing stages
run-all: bronze silver gold

# Default target
run: run-all
