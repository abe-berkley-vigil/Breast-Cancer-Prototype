# tests/test_predict.py

import os
from pathlib import Path
import pickle

# Import project modules using the same pattern as your other tests
import sys
import tempfile

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

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

# Import the functions to test

# Import data loading functions
from breast_cancer_classification.config import RAW_DATA_DIR
from breast_cancer_classification.dataset import load_data, preprocess_data
from breast_cancer_classification.modeling.predict import (
    evaluate_lr_model,
    generate_feature_importance,
    load_lr_model,
)


# Use the path from config
TEST_DATASET_PATH = Path(RAW_DATA_DIR) / "dataset.csv"


@pytest.fixture
def get_csv_file():
    """Use the actual dataset file."""
    return str(TEST_DATASET_PATH)


@pytest.fixture
def real_data(get_csv_file):
    """Load and preprocess the actual dataset."""
    if not Path(get_csv_file).exists():
        pytest.skip(f"Dataset file not found at {get_csv_file} - skipping tests requiring data")
    
    df = load_data(get_csv_file)
    return preprocess_data(df)


@pytest.fixture
def config():
    """Load intended Hydra configuration from project directory."""
    # Path to your actual config directory
    config_dir = package_dir / "conf"
    config_file = config_dir / "config.yaml"
    
    if not config_file.exists():
        pytest.skip(f"Config file not found at {config_file} - skipping tests requiring config")
    
    # Load the actual config file
    return OmegaConf.load(config_file)


@pytest.fixture
def trained_model_and_test_data(real_data, config):
    """Create a trained model and test data for prediction testing."""
    from breast_cancer_classification.modeling.train import (

        create_lr_model,
        create_test_train_split,
        fit_lr_model,
        scale_data,

    )
    
    # Split the data using config parameters
    X_train, X_test, y_train, y_test = create_test_train_split(
        real_data, 
        test_size=config.train.test_size, 
        random_state=config.train.random_state
    )
    
    # Scale data when configured
    if config.train.scale_data:
        X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    else:
        X_train_scaled, X_test_scaled = X_train, X_test
    
    # Create and train model using config parameters
    model = create_lr_model(max_iterv=config.model.lr_params.max_iter)
    fit_lr_model(model, X_train_scaled, y_train)
    
    return model, X_test_scaled, y_test





def test_evaluate_lr_model_with_data(trained_model_and_test_data):
    """Test model evaluation with real breast cancer data."""
    model, X_test, y_test = trained_model_and_test_data
    
    # Run evaluation
    y_pred, accuracy, conf_matrix, class_report = evaluate_lr_model(model, X_test, y_test)
    
    # Ensure shapes and types are correct
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(accuracy, float)
    assert isinstance(conf_matrix, np.ndarray)
    assert isinstance(class_report, str)
    
    # Check predictions shape matches test set
    assert len(y_pred) == len(y_test)
    
    # Check predictions are binary (0 or 1)
    assert all(pred in [0, 1] for pred in y_pred)
    
    # Check accuracy is reasonable for breast cancer data
    assert 0.0 <= accuracy <= 1.0
    
    # Check confusion matrix shape and values
    assert conf_matrix.shape == (2, 2)
    # Confusion matrix total should match number of test samples
    assert conf_matrix.sum() == len(y_test) 


def test_evaluate_lr_model_basic(trained_model_and_test_data):
    """Test basic model evaluation functionality."""
    model, X_test, y_test = trained_model_and_test_data
    
    # Run evaluation
    y_pred, accuracy, conf_matrix, class_report = evaluate_lr_model(model, X_test, y_test)
    
    # Check returned items of model are correct types
    assert isinstance(y_pred, np.ndarray)
    assert len(y_pred) == len(y_test)
    assert 0.0 <= accuracy <= 1.0
    assert isinstance(class_report, str)
    assert "precision" in class_report.lower()
    assert "recall" in class_report.lower()



def test_generate_feature_importance_with_data(trained_model_and_test_data, real_data):
    """Test feature importance generation with real breast cancer data."""
    model, _, _ = trained_model_and_test_data
    
    # Use the original dataset for feature names
    X_original = real_data.drop(["diagnosis", "id"], axis=1)
    
    # Generate feature importance
    importance_df = generate_feature_importance(model, X_original)
    
    # Check return type and structure
    assert isinstance(importance_df, pd.DataFrame)
    
    # Check required columns exist
    required_columns = ["feature", "coefficient", "abs_coefficient"]
    for col in required_columns:
        assert col in importance_df.columns
    
    # Check number of features matches model
    assert len(importance_df) == model.coef_.shape[1]
    
    # Check that features are sorted by absolute coefficient (descending)
    abs_coeffs = importance_df["abs_coefficient"].values
    assert all(abs_coeffs[i] >= abs_coeffs[i+1] for i in range(len(abs_coeffs)-1))
    
    # Check that absolute coefficients are non-negative
    assert all(val >= 0 for val in importance_df["abs_coefficient"])
    
    # Check that abs_coefficient matches the absolute value of coefficient
    for _, row in importance_df.iterrows():
        assert abs(row["coefficient"]) == row["abs_coefficient"]




def test_load_lr_model_success(trained_model_and_test_data):
    """Test successful model loading from file."""
    model, _, _ = trained_model_and_test_data
    
    # Save model to temporary file
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    
    try:
        # Save the model
        with open(tmp_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Load the model using related function
        loaded_model = load_lr_model(tmp_path)
        
        # Check that proper model type is returned
        assert isinstance(loaded_model, LogisticRegression)
        
        # Ensure that model parameters are preserved
        assert loaded_model.max_iter == model.max_iter
        np.testing.assert_array_equal(loaded_model.coef_, model.coef_)
        np.testing.assert_array_equal(loaded_model.intercept_, model.intercept_)
        
    finally:
        # Clean up
        if tmp_path.exists():
            os.unlink(tmp_path)


def test_load_lr_model_file_not_found():
    """Test error handling when model file doesn't exist."""
    non_existent_path = Path("non_existent_model.pkl")
    
    with pytest.raises(FileNotFoundError):
        load_lr_model(non_existent_path)


def test_load_lr_model_invalid_file():
    """Test error handling when file exists but contains invalid data."""
    # Create file with invalid content
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        tmp_file.write(b"not a pickled model")
    
    try:
        with pytest.raises(Exception):  # Should raise pickle or other loading error
            load_lr_model(tmp_path)
    finally:
        if tmp_path.exists():
            os.unlink(tmp_path)

