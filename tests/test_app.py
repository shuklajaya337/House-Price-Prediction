import os
import sys
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ─── Mock Streamlit before importing main ──────────────────────────────────────
# This prevents Streamlit GUI calls and caching decorators from raising errors.
def dummy_decorator(func):
    return func

from unittest.mock import MagicMock
mock_st = MagicMock()
mock_st.cache_data = dummy_decorator

# Define helper functions to return list of mocks to support unpacking
def mock_columns(n):
    if isinstance(n, int):
        return [MagicMock() for _ in range(n)]
    return [MagicMock() for _ in range(len(n))]

def mock_tabs(tab_list):
    return [MagicMock() for _ in range(len(tab_list))]

# Define helper functions to return realistic values for sliders and inputs
def mock_slider(label, min_value=None, max_value=None, value=None, *args, **kwargs):
    if value is not None:
        return value
    if len(args) >= 1: # standard positional args check
        return args[0]
    return 1.0

def mock_selectbox(label, options, *args, **kwargs):
    return options[0] if options else ""

mock_st.columns = mock_columns
mock_st.tabs = mock_tabs
mock_st.slider = mock_slider
mock_st.selectbox = mock_selectbox
mock_st.button.return_value = False

sys.modules['streamlit'] = mock_st

# Add root folder to sys.path to resolve src imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our ML functions from the src directory
from src.main import load_and_train


def test_dataset_exists():
    """Verify that the dataset is in the correct directory."""
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "housing_new.csv"
    )
    assert os.path.exists(data_path), f"Dataset not found at {data_path}"


def test_data_loading_and_cleaning():
    """Test that the data is loaded and cleaned correctly (no NaNs)."""
    model, scaler, feature_cols, X_test, y_test, y_pred, rmse, r2, data = load_and_train()
    
    assert data is not None
    assert len(data) > 0
    assert data.isnull().sum().sum() == 0, "Cleaned dataset contains NaN values"


def test_feature_columns_and_encoding():
    """Assert that ocean proximity is one-hot encoded and columns match expectations."""
    model, scaler, feature_cols, X_test, y_test, y_pred, rmse, r2, data = load_and_train()
    
    # Check that ocean_proximity columns exist in one-hot encoded form
    encoded_cols = [c for c in feature_cols if 'ocean_proximity_' in c]
    assert len(encoded_cols) > 0, "Ocean proximity features were not one-hot encoded"
    
    # Check shape compatibility
    assert X_test.shape[1] == len(feature_cols)


def test_model_performance():
    """Verify that the model meets basic performance benchmarks (R² > 0.70)."""
    model, scaler, feature_cols, X_test, y_test, y_pred, rmse, r2, data = load_and_train()
    
    assert r2 > 0.70, f"Model performance dropped significantly. R²: {r2}"
    assert rmse > 0, "Root Mean Squared Error must be a positive number"


def test_prediction_output():
    """Verify that model predictions are within reasonable bounds."""
    model, scaler, feature_cols, X_test, y_test, y_pred, rmse, r2, data = load_and_train()
    
    # Pick a sample row
    sample_input = X_test.iloc[0:1]
    sample_sc = scaler.transform(sample_input)
    prediction = model.predict(sample_sc)[0]
    
    # Check that prediction is positive and numeric
    assert isinstance(prediction, float)
    assert prediction > 0
    # California house median values are typically between 10k and 500k+
    assert 10000 <= prediction <= 600000
