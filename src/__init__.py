"""
Solid ML Stack - A comprehensive machine learning pipeline

This package provides a complete ML pipeline including:
- Feature engineering
- Machine learning models and training
- Preprocessing and optimization
- Evaluation and submission
- Utilities and configuration
"""

from . import features
from . import modeling
from . import preprocessing
from . import optimization
from . import evaluation
from . import submission
from . import utils

__version__ = "0.1.0"
__all__ = ['features', 'modeling', 'preprocessing', 'optimization', 'evaluation', 'submission', 'utils'] 