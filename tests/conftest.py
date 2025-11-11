"""Shared pytest fixtures for GHSOM tests."""

import numpy as np
import pytest
from typing import Dict, Any


@pytest.fixture
def sample_dataset():
    """Small synthetic dataset for quick tests (50 samples, 10 features)."""
    np.random.seed(42)
    return np.random.rand(50, 10)


@pytest.fixture
def large_dataset():
    """Larger dataset for integration tests (1000 samples, 20 features)."""
    np.random.seed(42)
    return np.random.rand(1000, 20)


@pytest.fixture
def high_dim_dataset():
    """High-dimensional dataset for stress tests (100 samples, 1000 features)."""
    np.random.seed(42)
    return np.random.rand(100, 1000)


@pytest.fixture
def tiny_dataset():
    """Minimal dataset for edge case testing (2 samples, 5 features)."""
    np.random.seed(42)
    return np.random.rand(2, 5)


@pytest.fixture
def single_point_dataset():
    """Single data point for edge case testing."""
    np.random.seed(42)
    return np.random.rand(1, 10)


@pytest.fixture
def ghsom_default_config() -> Dict[str, Any]:
    """Standard GHSOM configuration with default parameters."""
    return {
        "t1": 0.5,
        "t2": 0.05,
        "learning_rate": 0.1,
        "decay": 0.9,
        "gaussian_sigma": 1.0,
        "growing_metric": "qe",
    }


@pytest.fixture
def ghsom_shallow_config() -> Dict[str, Any]:
    """Configuration for shallow hierarchies (larger t1, larger t2)."""
    return {
        "t1": 0.3,
        "t2": 0.1,
        "learning_rate": 0.1,
        "decay": 0.9,
        "gaussian_sigma": 1.0,
        "growing_metric": "qe",
    }


@pytest.fixture
def ghsom_deep_config() -> Dict[str, Any]:
    """Configuration for deep hierarchies (larger t1, smaller t2)."""
    return {
        "t1": 0.7,
        "t2": 0.01,
        "learning_rate": 0.1,
        "decay": 0.9,
        "gaussian_sigma": 1.0,
        "growing_metric": "qe",
    }


@pytest.fixture
def mock_callback():
    """Mock callback for testing callback interface."""
    from ghsom.callbacks.base import TrackingCallback

    class MockCallback(TrackingCallback):
        def __init__(self):
            self.train_begin_called = False
            self.map_created_called = False
            self.train_end_called = False
            self.config = None
            self.metrics_log = []
            self.results = None

        def on_train_begin(self, config):
            self.train_begin_called = True
            self.config = config

        def on_map_created(self, metrics):
            self.map_created_called = True
            self.metrics_log.append(metrics)

        def on_train_end(self, results):
            self.train_end_called = True
            self.results = results

    return MockCallback()


@pytest.fixture
def neuron_builder():
    """Standard neuron builder for testing."""
    from ghsom.builders import NeuronBuilder

    builder = NeuronBuilder(tau_2=0.05, growing_metric="qe")
    builder.zero_quantization_error = 0.1
    return builder


@pytest.fixture
def sample_weights_2x2():
    """2x2 weight map for GSOM testing."""
    np.random.seed(42)
    return np.random.rand(2, 2, 10)


@pytest.fixture
def sample_weights_3x3():
    """3x3 weight map for GSOM testing."""
    np.random.seed(42)
    return np.random.rand(3, 3, 10)
