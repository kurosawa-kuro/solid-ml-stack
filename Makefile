# Makefile for solid-ml-stack

.PHONY: import-raw-data
import-raw-data:
	python3 src/import-raw-data.py

# Default target
run: import-raw-data
