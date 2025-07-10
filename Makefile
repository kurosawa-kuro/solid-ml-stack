# Makefile for Solid ML Stack

.PHONY: install dev-install test lint format clean help
.PHONY: preprocess features model optimize ensemble submission
.PHONY: kaggle-workflow kaggle-regression kaggle-classification kaggle-help
.PHONY: quick-run full-workflow benchmark

# Installation targets
install:
	pip install -e .

dev-install:
	pip install -e .[dev,optimization,visualization]

install-optimization:
	pip install -e .[optimization]

install-visualization:
	pip install -e .[visualization]

# Development targets
test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

test-unit:
	pytest tests/test_preprocessing.py tests/test_features.py tests/test_modeling.py tests/test_submission.py -v

test-integration:
	pytest tests/test_integration.py -v -m integration

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=60

test-smoke:
	pytest tests/test_integration.py::TestCLISmoke -v

lint:
	black src/ tests/ scripts/ --check
	flake8 src/ tests/ scripts/
	mypy src/

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

# Data processing targets
preprocess:
	python -c "from src.preprocessing import Preprocessor; print('Preprocessing module ready')"

features:
	python -c "from src.features.engineering.pipeline import FeatureEngineeringPipeline; print('Feature engineering module ready')"

# Model training targets
model-xgb:
	python -c "from src.modeling import ModelFactory; factory = ModelFactory(); model = factory.create_model_from_name('xgboost'); print(f'XGBoost model created: {model.config.name}')"

model-lgb:
	python -c "from src.modeling import ModelFactory; factory = ModelFactory(); model = factory.create_model_from_name('lightgbm'); print(f'LightGBM model created: {model.config.name}')"

model-cat:
	python -c "from src.modeling import ModelFactory; factory = ModelFactory(); model = factory.create_model_from_name('catboost'); print(f'CatBoost model created: {model.config.name}')"

# Optimization targets
optimize-grid:
	python -c "from src.optimization import GridSearchOptimizer; print('Grid search optimizer ready')"

optimize-random:
	python -c "from src.optimization import RandomSearchOptimizer; print('Random search optimizer ready')"

optimize-bayesian:
	python -c "from src.optimization import BayesianOptimizer; print('Bayesian optimizer ready')"

optimize-optuna:
	python -c "from src.optimization import OptunaOptimizer; print('Optuna optimizer ready')"

# Ensemble targets
ensemble-average:
	python -c "from src.modeling.ensemble import AverageEnsemble; print('Average ensemble ready')"

ensemble-stacking:
	python -c "from src.modeling.ensemble import StackingEnsemble; print('Stacking ensemble ready')"

ensemble-voting:
	python -c "from src.modeling.ensemble import VotingEnsemble; print('Voting ensemble ready')"

# Submission targets
submission:
	python -c "from src.submission import SubmissionGenerator; gen = SubmissionGenerator(); print('Submission generator ready')"

# Kaggle workflow targets
kaggle-workflow:
	@echo "Usage: make kaggle-workflow TRAIN=path/to/train.csv TEST=path/to/test.csv TARGET=target_column"
	@echo "Example: make kaggle-workflow TRAIN=data/raw/train.csv TEST=data/raw/test.csv TARGET=price"

kaggle-workflow-run:
	python scripts/kaggle_workflow.py \
		--train-path $(TRAIN) \
		--test-path $(TEST) \
		--target-col $(TARGET) \
		--problem-type $(TYPE) \
		--output-dir outputs

kaggle-regression:
	python scripts/kaggle_workflow.py \
		--train-path $(TRAIN) \
		--test-path $(TEST) \
		--target-col $(TARGET) \
		--problem-type regression \
		--output-dir outputs \
		--optimize \
		--ensemble

kaggle-classification:
	python scripts/kaggle_workflow.py \
		--train-path $(TRAIN) \
		--test-path $(TEST) \
		--target-col $(TARGET) \
		--problem-type classification \
		--output-dir outputs \
		--optimize \
		--ensemble

# Quick demo targets
quick-run:
	@echo "Running quick demo with house data..."
	python scripts/kaggle_workflow.py \
		--train-path data/raw/house_data.csv \
		--test-path data/raw/house_data.csv \
		--target-col price \
		--problem-type regression \
		--output-dir outputs/demo

full-workflow:
	@echo "Running full workflow with optimization..."
	python scripts/kaggle_workflow.py \
		--train-path $(TRAIN) \
		--test-path $(TEST) \
		--target-col $(TARGET) \
		--problem-type $(TYPE) \
		--output-dir outputs \
		--optimize \
		--ensemble

# Benchmarking targets
benchmark-models:
	python -c "\
	from src.modeling.factory import ModelFactory; \
	factory = ModelFactory(); \
	models = factory.get_default_models('regression'); \
	print(f'Created {len(models)} benchmark models'); \
	for m in models: print(f'  - {m.config.name}'); \
	"

benchmark-optimizers:
	python -c "\
	from src.optimization.factory import OptimizerFactory; \
	factory = OptimizerFactory(); \
	optimizers = factory.get_available_optimizers(); \
	print(f'Available optimizers: {optimizers}'); \
	"

# Setup and cleanup
setup:
	mkdir -p data/raw data/processed outputs submissions logs
	touch data/raw/.gitkeep data/processed/.gitkeep outputs/.gitkeep submissions/.gitkeep logs/.gitkeep

clean:
	rm -rf outputs/* submissions/* logs/*
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

clean-all: clean
	rm -rf data/processed/*

# Docker targets (future)
docker-build:
	@echo "Docker support coming soon..."

docker-run:
	@echo "Docker support coming soon..."

# Documentation targets
docs:
	@echo "Generating documentation..."
	@echo "Available modules:"
	@echo "  - preprocessing: Data preprocessing and cleaning"
	@echo "  - features.engineering: Feature engineering and generation"
	@echo "  - modeling: Model training and evaluation"
	@echo "  - optimization: Hyperparameter optimization"
	@echo "  - evaluation: Model evaluation metrics"
	@echo "  - submission: Submission file generation"

# Configuration examples
config-regression:
	python -c "\
	from config.kaggle_config import ConfigPresets; \
	config = ConfigPresets.regression_competition(); \
	print('Regression config created'); \
	print(f'Problem type: {config.problem_type}'); \
	"

config-classification:
	python -c "\
	from config.kaggle_config import ConfigPresets; \
	config = ConfigPresets.classification_competition(); \
	print('Classification config created'); \
	print(f'Problem type: {config.problem_type}'); \
	"

# Help targets
kaggle-help:
	@echo "Solid ML Stack - Competition-focused ML Pipeline"
	@echo ""
	@echo "Quick Start:"
	@echo "  make install              - Install basic dependencies"
	@echo "  make dev-install          - Install with dev tools and optimizers"
	@echo "  make setup               - Create necessary directories"
	@echo ""
	@echo "Kaggle Workflow:"
	@echo "  make kaggle-regression TRAIN=train.csv TEST=test.csv TARGET=price"
	@echo "  make kaggle-classification TRAIN=train.csv TEST=test.csv TARGET=class"
	@echo "  make quick-run           - Demo with house data"
	@echo ""
	@echo "Individual Components:"
	@echo "  make preprocess          - Test preprocessing module"
	@echo "  make features            - Test feature engineering"
	@echo "  make model-xgb           - Test XGBoost model"
	@echo "  make ensemble-stacking   - Test stacking ensemble"
	@echo "  make optimize-optuna     - Test Optuna optimizer"
	@echo ""
	@echo "Development:"
	@echo "  make test                - Run tests"
	@echo "  make lint                - Check code quality"
	@echo "  make format              - Format code"
	@echo "  make clean               - Clean generated files"
	@echo ""
	@echo "Examples:"
	@echo "  # Full workflow with optimization"
	@echo "  make kaggle-regression TRAIN=data/train.csv TEST=data/test.csv TARGET=SalePrice"
	@echo ""
	@echo "  # Classification competition"
	@echo "  make kaggle-classification TRAIN=data/train.csv TEST=data/test.csv TARGET=Survived"

help: kaggle-help

# Default target
.DEFAULT_GOAL := help

# Environment variables with defaults
TRAIN ?= data/raw/train.csv
TEST ?= data/raw/test.csv
TARGET ?= target
TYPE ?= regression
OUTPUT_DIR ?= outputs