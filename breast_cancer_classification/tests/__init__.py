# tests/__init__.py

import os
from pathlib import Path
import sys

# Get the correct directory structure
TESTS_DIR = Path(__file__).parent              # .../breast_cancer_classification/tests
PACKAGE_DIR = TESTS_DIR.parent                 # .../breast_cancer_classification
PROJECT_ROOT = PACKAGE_DIR.parent              # .../Breast-Cancer-Classification---SE489

# Add project root to Python path so we can import breast_cancer_classification
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set environment variable for config.py if it needs it
if "PROJ_ROOT" not in os.environ:
    os.environ["PROJ_ROOT"] = str(PROJECT_ROOT)

# Define test data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Common test data files
TEST_DATASET_PATH = RAW_DATA_DIR / "dataset.csv"

# Make these available to all test files
__all__ = [
    'TESTS_DIR',
    'PACKAGE_DIR',
    'PROJECT_ROOT', 
    'DATA_DIR',
    'RAW_DATA_DIR',
    'PROCESSED_DATA_DIR',
    'INTERIM_DATA_DIR',
    'EXTERNAL_DATA_DIR',
    'TEST_DATASET_PATH'
]