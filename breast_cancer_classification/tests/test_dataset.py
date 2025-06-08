
import os
from pathlib import Path
import sys

import pytest


# Get the correct paths
current_file = Path(__file__).resolve()
tests_dir = current_file.parent  # .../breast_cancer_classification/tests
package_dir = tests_dir.parent    # .../breast_cancer_classification
project_root = package_dir.parent # .../Breast-Cancer-Classification---SE489

# Add the project root to sys.path so we can import breast_cancer_classification
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set environment variable for config.py
os.environ["PROJ_ROOT"] = str(project_root)

# Now we can import!
from breast_cancer_classification.config import RAW_DATA_DIR
from breast_cancer_classification.dataset import load_data, preprocess_data

# Use the path from config
TEST_DATASET_PATH = Path(RAW_DATA_DIR) / "dataset.csv"

@pytest.fixture
def get_csv_file():
    """Use the actual dataset file."""
    return str(TEST_DATASET_PATH)

@pytest.fixture
def get_preprocessed_data():
    """Load and preprocess the dataset."""
    df = load_data(get_csv_file())
    return preprocess_data(df)

def test_load_data_shape(get_csv_file):
    """Test that loaded data is 2-dimensional."""
    df = load_data(get_csv_file)
    assert df.shape[0] > 0, f"Verify data has rows, received {df.shape[0]}"
    assert df.shape[1] > 0, f"Verify data has columns, received {df.shape[1]}"

def test_load_data_has_diagnosis_column(get_csv_file):
    """Test that loaded data has response variable."""
    df = load_data(get_csv_file)
    assert 'diagnosis' in df.columns, f"Expected 'diagnosis' column in {list(df.columns)}"

def test_load_data_no_missing_values(get_csv_file):
    """Test that data file has no missing values."""
    df = load_data(get_csv_file)
    missing_count = df.isnull().sum().sum()
    assert missing_count == 0, f"Expected 0 missing values, received {missing_count} missing values"

def test_preprocess_diagnosis_mapping(get_preprocessed_data):
    """Test that diagnosis values are correctly mapped."""
    df = get_preprocessed_data
    unique_values = set(df['diagnosis'].unique())
    EXPECTED_DIAGNOSIS_VALUES = {0, 1}  # Assuming 'B' is mapped to 0 and 'M' is mapped to 1
    assert unique_values == EXPECTED_DIAGNOSIS_VALUES, f"Expected {EXPECTED_DIAGNOSIS_VALUES} values, got {unique_values} unique values in 'diagnosis' column"

