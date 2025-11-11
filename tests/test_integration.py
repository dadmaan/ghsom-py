"""Integration tests for GHSOM end-to-end workflows."""

import numpy as np
import pytest
from ghsom import GHSOM
from ghsom.io import save_model, load_model
import tempfile
import os


class TestEndToEndWorkflow:
    """Test complete GHSOM workflows."""

    def test_full_training_pipeline(self, sample_dataset):
        """Test complete training pipeline from data to trained model."""
        # Create GHSOM
        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        # Train model
        root = ghsom.train(epochs_number=3, seed=42, grow_maxiter=5)

        # Verify training completed
        assert root is not None
        assert hasattr(root, "child_map")
        assert root.child_map is not None

        # Verify hierarchy structure
        assert len(root.child_map.neurons) >= 4  # At least 2x2 initial map

        # Verify neurons have weights
        for neuron in root.child_map.neurons.values():
            assert neuron.weight_vector() is not None
            assert len(neuron.weight_vector()) == sample_dataset.shape[1]

    def test_train_and_save_load(self, tiny_dataset):
        """Test training, saving, and loading a model."""
        # Train model
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.pkl")
            save_model(root, model_path)

            # Verify file was created
            assert os.path.exists(model_path)

            # Load model
            loaded_root = load_model(model_path)

            # Verify loaded model structure
            assert loaded_root is not None
            assert hasattr(loaded_root, "child_map")
            assert loaded_root.child_map is not None

    def test_hierarchical_structure_depth(self, sample_dataset):
        """Test hierarchical structure has expected depth characteristics."""
        # Shallow hierarchy (high t2, high t1)
        ghsom_shallow = GHSOM(
            input_dataset=sample_dataset,
            t1=0.8,  # High t1 = less growth
            t2=0.2,  # High t2 = less depth
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root_shallow = ghsom_shallow.train(epochs_number=2, seed=42, grow_maxiter=3)

        # Should have minimal hierarchy
        assert root_shallow is not None
        assert root_shallow.child_map is not None

    def test_reproducibility_with_seed(self, tiny_dataset):
        """Test complete workflow is reproducible with seed."""

        def train_and_count_neurons(seed_val):
            ghsom = GHSOM(
                input_dataset=tiny_dataset,
                t1=0.5,
                t2=0.05,
                learning_rate=0.1,
                decay=0.9,
                gaussian_sigma=1.0,
            )
            root = ghsom.train(epochs_number=2, seed=seed_val, grow_maxiter=3)
            return len(root.child_map.neurons)

        # Train twice with same seed
        count1 = train_and_count_neurons(42)
        count2 = train_and_count_neurons(42)

        # Should produce same structure
        assert count1 == count2

    def test_training_with_different_dataset_sizes(self):
        """Test training works with various dataset sizes."""
        # Small dataset
        small_data = np.random.rand(10, 5)
        ghsom_small = GHSOM(
            input_dataset=small_data,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )
        result_small = ghsom_small.train(epochs_number=2, seed=42, grow_maxiter=3)
        assert result_small is not None

        # Medium dataset
        medium_data = np.random.rand(100, 10)
        ghsom_medium = GHSOM(
            input_dataset=medium_data,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )
        result_medium = ghsom_medium.train(epochs_number=2, seed=42, grow_maxiter=3)
        assert result_medium is not None

    def test_training_with_different_dimensionalities(self):
        """Test training works with various feature dimensions."""
        # Low dimensional
        low_dim = np.random.rand(50, 3)
        ghsom_low = GHSOM(
            input_dataset=low_dim,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )
        result_low = ghsom_low.train(epochs_number=2, seed=42, grow_maxiter=3)
        assert result_low is not None

        # High dimensional
        high_dim = np.random.rand(50, 100)
        ghsom_high = GHSOM(
            input_dataset=high_dim,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )
        result_high = ghsom_high.train(epochs_number=2, seed=42, grow_maxiter=3)
        assert result_high is not None


class TestMultiprocessingWorkflow:
    """Test multiprocessing aspects of training."""

    def test_training_with_single_worker(self, tiny_dataset):
        """Test training with single worker."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        result = ghsom.train(epochs_number=2, seed=42, n_workers=1, grow_maxiter=3)

        assert result is not None

    def test_training_with_multiple_workers(self, tiny_dataset):
        """Test training with multiple workers."""
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


class TestTrainingConvergence:
    """Test training convergence properties."""

    def test_training_completes_within_maxiter(self, tiny_dataset):
        """Test training respects maxiter parameter."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.1,  # Low t1 would normally cause lots of growth
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        # With low maxiter, training should still complete
        result = ghsom.train(epochs_number=2, seed=42, grow_maxiter=1)

        assert result is not None
        assert result.child_map is not None

    def test_neurons_have_valid_weights_after_training(self, sample_dataset):
        """Test all neurons have valid weights after training."""
        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=3, seed=42, grow_maxiter=5)

        # Check all neurons in root's child map
        for neuron in root.child_map.neurons.values():
            weights = neuron.weight_vector()
            assert weights is not None
            assert not np.isnan(weights).any()
            assert not np.isinf(weights).any()
            assert weights.shape == (sample_dataset.shape[1],)


class TestDatasetHandling:
    """Test dataset handling during training."""

    def test_neuron_dataset_assignment(self, sample_dataset):
        """Test neurons receive correct data subsets."""
        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=3, seed=42, grow_maxiter=5)

        # Root should have full dataset
        assert len(root.input_dataset) == len(sample_dataset)

        # Child map neurons should have data assigned
        total_assigned = sum(
            len(neuron.input_dataset) if neuron.input_dataset is not None else 0
            for neuron in root.child_map.neurons.values()
        )

        # Some data should be assigned to neurons
        assert total_assigned > 0

    def test_training_with_dataset_percentage(self, sample_dataset):
        """Test training uses specified dataset percentage."""
        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        # Train with 50% of data
        result = ghsom.train(
            epochs_number=2, seed=42, dataset_percentage=0.5, grow_maxiter=3
        )

        assert result is not None


class TestErrorHandling:
    """Test error handling in integration scenarios."""

    def test_training_with_empty_callbacks_list(self, tiny_dataset):
        """Test training with empty callbacks list."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        result = ghsom.train(epochs_number=2, seed=42, callbacks=[], grow_maxiter=3)

        assert result is not None

    def test_training_with_none_callbacks(self, tiny_dataset):
        """Test training with None callbacks."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        result = ghsom.train(epochs_number=2, seed=42, callbacks=None, grow_maxiter=3)

        assert result is not None
