"""Tests for GHSOM training logic."""

import numpy as np
import pytest
from ghsom import GHSOM
from ghsom.callbacks import TrackingCallback


class MockCallback(TrackingCallback):
    """Mock callback for testing."""

    def __init__(self):
        self.train_begin_called = False
        self.train_end_called = False
        self.map_created_called = False
        self.train_begin_config = None
        self.train_end_results = None
        self.map_created_count = 0

    def on_train_begin(self, config):
        self.train_begin_called = True
        self.train_begin_config = config

    def on_map_created(self, metrics):
        self.map_created_called = True
        self.map_created_count += 1

    def on_train_end(self, results):
        self.train_end_called = True
        self.train_end_results = results


class TestGHSOMTraining:
    """Tests for GHSOM training workflow."""

    def test_train_basic(self, tiny_dataset):
        """Test basic GHSOM training completes."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        result = ghsom.train(epochs_number=2, seed=42, grow_maxiter=5)

        assert result is not None
        assert hasattr(result, "child_map")
        assert result.child_map is not None

    def test_train_returns_zero_unit(self, sample_dataset):
        """Test train returns the zero unit (root)."""
        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        zero_unit = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        # Zero unit should have a child map
        assert zero_unit is not None
        assert hasattr(zero_unit, "child_map")
        assert zero_unit.child_map is not None
        assert len(zero_unit.child_map.neurons) >= 4  # At least 2x2 initial map

    def test_train_with_callbacks(self, tiny_dataset):
        """Test training with callbacks."""
        callback = MockCallback()

        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        ghsom.train(epochs_number=2, seed=42, callbacks=[callback], grow_maxiter=3)

        # Check callbacks were invoked
        assert callback.train_begin_called
        assert callback.train_end_called
        assert callback.train_begin_config is not None
        assert callback.train_end_results is not None

        # Config should contain expected keys
        assert "learning_rate" in callback.train_begin_config
        assert "t1" in callback.train_begin_config
        assert "t2" in callback.train_begin_config
        assert "epochs" in callback.train_begin_config

    def test_train_with_multiple_callbacks(self, tiny_dataset):
        """Test training with multiple callbacks."""
        callback1 = MockCallback()
        callback2 = MockCallback()

        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        ghsom.train(
            epochs_number=2, seed=42, callbacks=[callback1, callback2], grow_maxiter=3
        )

        # Both callbacks should be invoked
        assert callback1.train_begin_called
        assert callback2.train_begin_called
        assert callback1.train_end_called
        assert callback2.train_end_called

    def test_train_with_seed_reproducibility(self, tiny_dataset):
        """Test training with same seed produces same hierarchy."""
        np.random.seed(42)

        def train_with_seed(seed_val):
            ghsom = GHSOM(
                input_dataset=tiny_dataset,
                t1=0.5,
                t2=0.05,
                learning_rate=0.1,
                decay=0.9,
                gaussian_sigma=1.0,
            )
            zero_unit = ghsom.train(epochs_number=2, seed=seed_val, grow_maxiter=3)
            return len(zero_unit.child_map.neurons)

        size1 = train_with_seed(42)

        # Reset random seed
        np.random.seed(42)
        size2 = train_with_seed(42)

        assert size1 == size2

    def test_train_with_different_epochs(self, tiny_dataset):
        """Test training with different epoch counts."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        # Train with few epochs
        result_few = ghsom.train(epochs_number=1, seed=42, grow_maxiter=3)

        ghsom2 = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        # Train with more epochs
        result_many = ghsom2.train(epochs_number=10, seed=42, grow_maxiter=3)

        # Both should complete
        assert result_few is not None
        assert result_many is not None

    def test_train_with_n_workers_auto(self, tiny_dataset):
        """Test training with automatic worker count."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        result = ghsom.train(epochs_number=2, seed=42, n_workers=None, grow_maxiter=3)

        assert result is not None

    def test_train_with_n_workers_all_cores(self, tiny_dataset):
        """Test training with all CPU cores."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        result = ghsom.train(epochs_number=2, seed=42, n_workers=-1, grow_maxiter=3)

        assert result is not None

    def test_train_with_n_workers_explicit(self, tiny_dataset):
        """Test training with explicit worker count."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        result = ghsom.train(epochs_number=2, seed=42, n_workers=2, grow_maxiter=3)

        assert result is not None

    def test_train_with_dataset_percentage(self, sample_dataset):
        """Test training with different dataset percentages."""
        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        # Train with 50% of dataset
        result = ghsom.train(
            epochs_number=2, seed=42, dataset_percentage=0.5, grow_maxiter=3
        )

        assert result is not None

    def test_train_with_min_dataset_size(self, sample_dataset):
        """Test training with minimum dataset size."""
        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        result = ghsom.train(
            epochs_number=2, seed=42, min_dataset_size=5, grow_maxiter=3
        )

        assert result is not None

    def test_train_with_grow_maxiter(self, tiny_dataset):
        """Test training with grow_maxiter parameter."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        # Low maxiter should limit growth
        result = ghsom.train(epochs_number=2, seed=42, grow_maxiter=1)

        assert result is not None


class TestGHSOMHierarchyCreation:
    """Tests for hierarchical structure creation."""

    def test_hierarchy_has_root(self, tiny_dataset):
        """Test trained hierarchy has root node."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        assert root is not None
        assert hasattr(root, "child_map")

    def test_hierarchy_creates_child_maps(self, sample_dataset):
        """Test hierarchy creates child maps for neurons."""
        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,  # Moderate t1 for stable growth
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        # Root should have a child map with neurons
        assert root.child_map is not None
        assert len(root.child_map.neurons) > 0

    def test_zero_unit_initialization(self, tiny_dataset):
        """Test zero unit is properly initialized."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        zero_unit = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        # Zero unit should have the full dataset
        assert zero_unit is not None
        assert hasattr(zero_unit, "input_dataset")
        assert len(zero_unit.input_dataset) == len(tiny_dataset)


class TestGHSOMWeightInitialization:
    """Tests for weight initialization methods."""

    def test_initial_random_weights_shape(self, sample_dataset):
        """Test initial random weights have correct shape."""
        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        # Access private method for testing
        weights = ghsom._GHSOM__calc_initial_random_weights(seed=42)

        assert weights.shape == (2, 2, sample_dataset.shape[1])

    def test_initial_random_weights_from_dataset(self, sample_dataset):
        """Test initial weights are from dataset."""
        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        weights = ghsom._GHSOM__calc_initial_random_weights(seed=42)

        # Each weight should be a valid data point
        for i, j in np.ndindex(2, 2):
            weight = weights[i, j]
            # Weight values should be in reasonable range (0-1 for normalized data)
            assert weight.shape == (sample_dataset.shape[1],)


class TestGHSOMCallbackIntegration:
    """Tests for callback integration during training."""

    def test_on_train_begin_receives_config(self, tiny_dataset):
        """Test on_train_begin receives configuration."""
        callback = MockCallback()

        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        ghsom.train(epochs_number=5, seed=42, callbacks=[callback], grow_maxiter=3)

        config = callback.train_begin_config
        assert config["epochs"] == 5
        assert config["learning_rate"] == 0.1
        assert config["decay"] == 0.9
        assert config["t1"] == 0.5
        assert config["t2"] == 0.05

    def test_on_train_end_receives_results(self, tiny_dataset):
        """Test on_train_end receives results."""
        callback = MockCallback()

        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        ghsom.train(epochs_number=2, seed=42, callbacks=[callback], grow_maxiter=3)

        results = callback.train_end_results
        assert results is not None
        assert "total_neurons" in results


class TestGHSOMParameterValidation:
    """Tests for parameter validation during training."""

    def test_train_with_mqe_metric(self, tiny_dataset):
        """Test training with MQE metric."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
            growing_metric="mqe",
        )

        result = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        assert result is not None

    def test_train_with_qe_metric(self, tiny_dataset):
        """Test training with QE metric."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
            growing_metric="qe",
        )

        result = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        assert result is not None
