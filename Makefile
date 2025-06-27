# Makefile for solid-ml-stack

.PHONY: import-raw-data
import-raw-data:
	python3 src/bronze.py

# Default target
run: import-raw-data
