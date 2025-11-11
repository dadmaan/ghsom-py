"""Comprehensive tests for GSOM class."""

import numpy as np
import pytest
from ghsom.core import GSOM
from ghsom.builders import NeuronBuilder


@pytest.fixture
def gsom_basic(sample_dataset, neuron_builder):
    """Create a basic GSOM for testing."""
    weights_map = np.random.rand(2, 2, 10)

    gsom = GSOM(
        initial_map_size=(2, 2),
        parent_quantization_error=1.0,
        t1=0.5,
        data_size=sample_dataset.shape[1],
        weights_map=weights_map,
        parent_dataset=sample_dataset,
        neuron_builder=neuron_builder,
    )
    return gsom


class TestGSOMInitialization:
    """Tests for GSOM initialization."""

    def test_gsom_basic_initialization(self, sample_dataset, neuron_builder):
        """Test basic GSOM creation."""
        weights_map = np.random.rand(2, 2, 10)

        gsom = GSOM(
            initial_map_size=(2, 2),
            parent_quantization_error=1.0,
            t1=0.5,
            data_size=sample_dataset.shape[1],
            weights_map=weights_map,
            parent_dataset=sample_dataset,
            neuron_builder=neuron_builder,
        )

        assert gsom is not None
        assert gsom.neurons is not None
        assert len(gsom.neurons) == 4  # 2x2 map

    def test_gsom_with_different_map_sizes(self, sample_dataset, neuron_builder):
        """Test GSOM with various map sizes."""
        map_sizes = [(2, 2), (3, 3), (2, 3), (4, 4)]

        for size in map_sizes:
            weights_map = np.random.rand(size[0], size[1], 10)

            gsom = GSOM(
                initial_map_size=size,
                parent_quantization_error=1.0,
                t1=0.5,
                data_size=sample_dataset.shape[1],
                weights_map=weights_map,
                parent_dataset=sample_dataset,
                neuron_builder=neuron_builder,
            )

            expected_neurons = size[0] * size[1]
            assert len(gsom.neurons) == expected_neurons

    def test_gsom_fails_with_empty_dataset(self, neuron_builder):
        """Test GSOM fails gracefully with empty dataset."""
        empty_data = np.empty((0, 10))
        weights_map = np.random.rand(2, 2, 10)

        with pytest.raises((AssertionError, ValueError)):
            GSOM(
                initial_map_size=(2, 2),
                parent_quantization_error=1.0,
                t1=0.5,
                data_size=0,
                weights_map=weights_map,
                parent_dataset=empty_data,
                neuron_builder=neuron_builder,
            )


class TestGSOMWinnerNeuron:
    """Tests for winner neuron selection."""

    def test_winner_neuron_single_input(self, gsom_basic, sample_dataset):
        """Test finding winner for single data point."""
        data_point = sample_dataset[0]

        winners, support = gsom_basic.winner_neuron(data_point)

        assert len(winners) == 1
        assert len(support) == len(gsom_basic.neurons)

    def test_winner_neuron_batch_input(self, gsom_basic, sample_dataset):
        """Test finding winners for multiple data points."""
        batch_data = sample_dataset[:10]

        winners, support = gsom_basic.winner_neuron(batch_data)

        assert len(winners) == 10
        # Each neuron should have a support list
        assert len(support) == len(gsom_basic.neurons)

    def test_winner_neuron_consistency(self, gsom_basic, sample_dataset):
        """Test winner selection is consistent."""
        data_point = sample_dataset[0]

        winners1, _ = gsom_basic.winner_neuron(data_point)
        winners2, _ = gsom_basic.winner_neuron(data_point)

        assert winners1[0].position == winners2[0].position

    def test_all_data_has_winner(self, gsom_basic, sample_dataset):
        """Test every data point gets assigned a winner."""
        winners, support = gsom_basic.winner_neuron(sample_dataset)

        # Total support should equal number of data points
        total_support = sum(len(s) for s in support)
        assert total_support == len(sample_dataset)


class TestGSOMTraining:
    """Tests for GSOM training."""

    def test_training_basic(self, sample_dataset, neuron_builder):
        """Test basic training completes without errors."""
        weights_map = np.random.rand(2, 2, 10)

        gsom = GSOM(
            initial_map_size=(2, 2),
            parent_quantization_error=10.0,
            t1=0.5,
            data_size=sample_dataset.shape[1],
            weights_map=weights_map,
            parent_dataset=sample_dataset,
            neuron_builder=neuron_builder,
        )

        result = gsom.train(
            epochs=3,
            initial_gaussian_sigma=1.0,
            initial_learning_rate=0.1,
            decay=0.9,
            dataset_percentage=1.0,
            min_dataset_size=1,
            seed=42,
            maxiter=5,
        )

        assert result is gsom
        assert len(gsom.neurons) >= 4  # Should have at least initial neurons

    def test_training_with_seed_reproducibility(self, sample_dataset, neuron_builder):
        """Test training with same seed produces same results."""
        np.random.seed(42)

        def train_gsom(seed_val):
            weights_map = np.random.rand(2, 2, 10)
            gsom = GSOM(
                initial_map_size=(2, 2),
                parent_quantization_error=10.0,
                t1=0.5,
                data_size=sample_dataset.shape[1],
                weights_map=weights_map.copy(),
                parent_dataset=sample_dataset,
                neuron_builder=neuron_builder,
            )
            gsom.train(
                epochs=2,
                initial_gaussian_sigma=1.0,
                initial_learning_rate=0.1,
                decay=0.9,
                dataset_percentage=1.0,
                min_dataset_size=1,
                seed=seed_val,
                maxiter=3,
            )
            return len(gsom.neurons)

        # Same seed should give same map size
        size1 = train_gsom(42)

        # Reset for second run
        np.random.seed(42)
        size2 = train_gsom(42)

        assert size1 == size2

    def test_training_epochs_effect(self, sample_dataset, neuron_builder):
        """Test different epoch counts."""
        weights_map = np.random.rand(2, 2, 10)

        gsom = GSOM(
            initial_map_size=(2, 2),
            parent_quantization_error=10.0,
            t1=0.5,
            data_size=sample_dataset.shape[1],
            weights_map=weights_map,
            parent_dataset=sample_dataset,
            neuron_builder=neuron_builder,
        )

        # Training with 1 epoch should complete
        result = gsom.train(
            epochs=1,
            initial_gaussian_sigma=1.0,
            initial_learning_rate=0.1,
            decay=0.9,
            dataset_percentage=1.0,
            min_dataset_size=1,
            seed=42,
            maxiter=2,
        )

        assert result is gsom


class TestGSOMGrowth:
    """Tests for GSOM map growth."""

    def test_map_can_grow(self, sample_dataset, neuron_builder):
        """Test that map can grow during training."""
        weights_map = np.random.rand(2, 2, 10)

        gsom = GSOM(
            initial_map_size=(2, 2),
            parent_quantization_error=100.0,  # High QE to trigger growth
            t1=0.1,  # Low t1 to encourage growth
            data_size=sample_dataset.shape[1],
            weights_map=weights_map,
            parent_dataset=sample_dataset,
            neuron_builder=neuron_builder,
        )

        initial_size = len(gsom.neurons)

        gsom.train(
            epochs=5,
            initial_gaussian_sigma=1.0,
            initial_learning_rate=0.1,
            decay=0.9,
            dataset_percentage=1.0,
            min_dataset_size=1,
            seed=42,
            maxiter=10,
        )

        final_size = len(gsom.neurons)

        # Map should grow (or at least not shrink)
        assert final_size >= initial_size

    def test_maxiter_limits_growth(self, sample_dataset, neuron_builder):
        """Test that maxiter limits training iterations."""
        weights_map = np.random.rand(2, 2, 10)

        gsom = GSOM(
            initial_map_size=(2, 2),
            parent_quantization_error=100.0,
            t1=0.1,
            data_size=sample_dataset.shape[1],
            weights_map=weights_map,
            parent_dataset=sample_dataset,
            neuron_builder=neuron_builder,
        )

        # With maxiter=1, training should stop quickly
        gsom.train(
            epochs=5,
            initial_gaussian_sigma=1.0,
            initial_learning_rate=0.1,
            decay=0.9,
            dataset_percentage=1.0,
            min_dataset_size=1,
            seed=42,
            maxiter=1,
        )

        # Should complete without infinite loop
        assert gsom is not None


class TestGSOMMapShape:
    """Tests for GSOM map shape evolution."""

    def test_initial_map_shape(self, sample_dataset, neuron_builder):
        """Test map has correct initial shape."""
        weights_map = np.random.rand(3, 3, 10)

        gsom = GSOM(
            initial_map_size=(3, 3),
            parent_quantization_error=1.0,
            t1=0.5,
            data_size=sample_dataset.shape[1],
            weights_map=weights_map,
            parent_dataset=sample_dataset,
            neuron_builder=neuron_builder,
        )

        # Should have 3x3=9 neurons
        assert len(gsom.neurons) == 9

    def test_neurons_have_correct_positions(self, sample_dataset, neuron_builder):
        """Test neurons are placed at correct positions."""
        weights_map = np.random.rand(2, 3, 10)

        gsom = GSOM(
            initial_map_size=(2, 3),
            parent_quantization_error=1.0,
            t1=0.5,
            data_size=sample_dataset.shape[1],
            weights_map=weights_map,
            parent_dataset=sample_dataset,
            neuron_builder=neuron_builder,
        )

        positions = [neuron.position for neuron in gsom.neurons.values()]

        # All positions should be unique
        assert len(positions) == len(set(positions))

        # Positions should be within map bounds
        for pos in positions:
            assert 0 <= pos[0] < 2
            assert 0 <= pos[1] < 3


class TestGSOMEdgeCases:
    """Edge case tests for GSOM."""

    def test_gsom_with_tiny_dataset(self, tiny_dataset, neuron_builder):
        """Test GSOM with very small dataset."""
        weights_map = np.random.rand(2, 2, 5)

        gsom = GSOM(
            initial_map_size=(2, 2),
            parent_quantization_error=1.0,
            t1=0.5,
            data_size=tiny_dataset.shape[1],
            weights_map=weights_map,
            parent_dataset=tiny_dataset,
            neuron_builder=neuron_builder,
        )

        assert gsom is not None
        assert len(gsom.neurons) == 4

    def test_gsom_with_high_dimensional_data(self, high_dim_dataset, neuron_builder):
        """Test GSOM with high-dimensional data."""
        weights_map = np.random.rand(2, 2, 1000)

        gsom = GSOM(
            initial_map_size=(2, 2),
            parent_quantization_error=1.0,
            t1=0.5,
            data_size=high_dim_dataset.shape[1],
            weights_map=weights_map,
            parent_dataset=high_dim_dataset,
            neuron_builder=neuron_builder,
        )

        assert gsom is not None

        # Test winner neuron with high-dim data
        winners, _ = gsom.winner_neuron(high_dim_dataset[0])
        assert len(winners) == 1

    def test_gsom_with_extreme_t1(self, sample_dataset, neuron_builder):
        """Test GSOM with extreme t1 values."""
        weights_map = np.random.rand(2, 2, 10)

        # Very high t1 (should limit growth)
        gsom = GSOM(
            initial_map_size=(2, 2),
            parent_quantization_error=1.0,
            t1=0.99,
            data_size=sample_dataset.shape[1],
            weights_map=weights_map,
            parent_dataset=sample_dataset,
            neuron_builder=neuron_builder,
        )

        assert gsom is not None

        # Very low t1 (should encourage growth)
        gsom2 = GSOM(
            initial_map_size=(2, 2),
            parent_quantization_error=1.0,
            t1=0.01,
            data_size=sample_dataset.shape[1],
            weights_map=weights_map.copy(),
            parent_dataset=sample_dataset,
            neuron_builder=neuron_builder,
        )

        assert gsom2 is not None
