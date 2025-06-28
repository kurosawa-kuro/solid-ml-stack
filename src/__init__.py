"""
Solid ML Stack - A comprehensive machine learning pipeline

This package provides a complete ML pipeline including:
- Data processing stages (bronze, silver, gold)
- Machine learning models and training
- Feature engineering
- Pipeline orchestration
- Utilities and configuration
"""

from . import ml
from . import data_stage
from . import pipelines
from . import utils
from . import features

__version__ = "0.1.0"
__all__ = ['ml', 'data_stage', 'pipelines', 'utils', 'features'] 