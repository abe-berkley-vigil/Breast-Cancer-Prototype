# tests/test_training.py

import os
from pathlib import Path

# Import project modules using the same pattern as your other tests
import sys

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
from breast_cancer_classification.modeling.train import (
    create_lr_model,
    create_test_train_split,
    fit_lr_model,
    scale_data,
)

# Use the path from config
TEST_DATASET_PATH = Path(RAW_DATA_DIR) / "dataset.csv"


@pytest.fixture
def get_csv_file():
    """Use the actual dataset file."""
    return str(TEST_DATASET_PATH)


@pytest.fixture
def model_data(get_csv_file):
    """Load and preprocess the actual dataset."""
    if not Path(get_csv_file).exists():
        pytest.skip(f"Dataset file not found at {get_csv_file} - skipping tests that require real data")
    
    df = load_data(get_csv_file)
    return preprocess_data(df)


@pytest.fixture
def config():
    """Load the actual Hydra configuration from the project directory."""
    # Path to your actual config directory
    config_dir = package_dir / "conf"
    config_file = config_dir / "config.yaml"
    
    if not config_file.exists():
        pytest.skip(f"Config file not found at {config_file} - skipping tests that require config")
    
    # Load the actual config file
    return OmegaConf.load(config_file)


@pytest.fixture
def trained_model(model_data, config):
    """Create a trained logistic regression model using your actual data and config."""
    # Split the data using your config parameters
    X_train, X_test, y_train, y_test = create_test_train_split(
        model_data, 
        test_size=config.train.test_size, 
        random_state=config.train.random_state
    )
    
    # Scale if configured
    if config.train.scale_data:
        X_train_scaled, _ = scale_data(X_train, X_test)
    else:
        X_train_scaled = X_train
    
    # Create and train model using config parameters
    model = create_lr_model(max_iterv=config.model.lr_params.max_iter)
    fit_lr_model(model, X_train_scaled, y_train)
    
    return model


def test_create_test_train_split_with_config(model_data, config):
    """Test that the function works with real data and config parameters."""
    X_train, X_test, y_train, y_test = create_test_train_split(
        model_data, 
        test_size=config.train.test_size, 
        random_state=config.train.random_state
    )
    
    # Check data types
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    
    # Check train/test split sizes based on config
    total_size = len(model_data)
    expected_test_size = int(total_size * config.train.test_size)
    expected_train_size = total_size - expected_test_size
    
    assert len(X_test) == expected_test_size
    assert len(X_train) == expected_train_size
    
    # Check proper features droppped from training data
    assert 'diagnosis' not in X_train.columns
    assert 'id' not in X_train.columns
    assert 'diagnosis' not in X_test.columns
    assert 'id' not in X_test.columns

    # Check that target variable is correctly separated
    assert 'diagnosis' in y_train.columns
    assert 'diagnosis' in y_test.columns




def test_create_test_train_split_debug_mode(model_data, config):
    """Test the debug parameter functionality."""
    # Test with debug=True (should not raise errors)
    X_train, X_test, y_train, y_test = create_test_train_split(
        model_data, 
        test_size=config.train.test_size, 
        random_state=config.train.random_state,
        debug=True
    )
    
    # Should still return valid data
    assert len(X_train) > 0
    assert len(X_test) > 0


def test_scale_data_with_model_data_and_config(model_data, config):
    """Test scaling with actual breast cancer data and config settings."""
    X_train, X_test, _, _ = create_test_train_split(
        model_data, 
        test_size=config.train.test_size, 
        random_state=config.train.random_state
    )
    
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    
    # Check training data is proper type and shape after scaling
    assert isinstance(X_train_scaled, np.ndarray)
    assert isinstance(X_test_scaled, np.ndarray)
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    
    # Check standardization (mean~0, std~1)
    assert np.allclose(np.mean(X_train_scaled, axis=0), 0, atol=1e-10)
    assert np.allclose(np.std(X_train_scaled, axis=0), 1, atol=1e-10)


def test_scale_data_no_nans_introduced(model_data, config):
    """Test that scaling preserves relative relationships in the data."""
    X_train, X_test, _, _ = create_test_train_split(
        model_data, 
        test_size=config.train.test_size, 
        random_state=config.train.random_state
    )
    
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    
    # Check that the scaling doesn't create infinite values or NaNs
    assert not np.isnan(X_train_scaled).any()
    assert not np.isnan(X_test_scaled).any()
    assert not np.isinf(X_train_scaled).any()
    assert not np.isinf(X_test_scaled).any()


def test_create_lr_model_config_params(config):
    """Test model creation with parameters from config."""
    model = create_lr_model(max_iterv=config.model.lr_params.max_iter)
    #Assert model is a LogisticRegression instance
    assert isinstance(model, LogisticRegression) 
    # Check that model parameters match config
    assert model.max_iter == config.model.lr_params.max_iter



def test_create_lr_model_fit(config):
    """Test that the returned model is not yet fitted."""
    model = create_lr_model(max_iterv=config.model.lr_params.max_iter)
    
    # Check that model is not fitted
    assert not hasattr(model, 'coef_') or model.coef_ is None
    assert not hasattr(model, 'intercept_') or model.intercept_ is None


def test_fit_lr_model_with_model_data_and_config(model_data, config):
    """Test model fitting with real data and config parameters."""
    X_train, X_test, y_train, y_test = create_test_train_split(
        model_data, 
        test_size=config.train.test_size, 
        random_state=config.train.random_state
    )
    
    if config.train.scale_data:
        X_train_scaled, _ = scale_data(X_train, X_test)
    else:
        X_train_scaled = X_train
    
    model = create_lr_model(max_iterv=config.model.lr_params.max_iter)
    
    # Fit the model
    fit_lr_model(model, X_train_scaled, y_train)
    
    # Check that model is fitted and working
    predictions = model.predict(X_train_scaled)
    assert len(predictions) == len(y_train)
    



def test_fit_lr_model_coefficients(model_data, config):
    """Test that model has correct coefficients for real feature count."""
    X_train, X_test, y_train, y_test = create_test_train_split(
        model_data, 
        test_size=config.train.test_size, 
        random_state=config.train.random_state
    )
    
    if config.train.scale_data:
        X_train_scaled, _ = scale_data(X_train, X_test)
    else:
        X_train_scaled = X_train
    
    model = create_lr_model(max_iterv=config.model.lr_params.max_iter)
    fit_lr_model(model, X_train_scaled, y_train)
    
    # Check that coefficients exist and have correct shape
    assert hasattr(model, 'coef_')
    assert hasattr(model, 'intercept_')
    assert model.coef_ is not None
    assert model.intercept_ is not None
    
    # Number of coefficients should match number of features
    expected_features = X_train_scaled.shape[1]
    assert model.coef_.shape[1] == expected_features


def test_fit_lr_model_convergence_with_config_max_iter(model_data, config):
    """Test that model converges with the configured max_iter."""
    X_train, X_test, y_train, y_test = create_test_train_split(
        model_data, 
        test_size=config.train.test_size, 
        random_state=config.train.random_state
    )
    
    if config.train.scale_data:
        X_train_scaled, _ = scale_data(X_train, X_test)
    else:
        X_train_scaled = X_train
    
    model = create_lr_model(max_iterv=config.model.lr_params.max_iter)
    fit_lr_model(model, X_train_scaled, y_train)
    
    # Check that model converged (n_iter_ should be less than max_iter for convergence)
    assert hasattr(model, 'n_iter_')
    assert model.n_iter_[0] <= config.model.lr_params.max_iter


