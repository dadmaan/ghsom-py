"""Tests for NeuronBuilder class."""

import numpy as np
import pytest
from ghsom.builders import NeuronBuilder
from ghsom.core import Neuron


class TestNeuronBuilderInitialization:
    """Tests for NeuronBuilder initialization."""

    def test_builder_basic_initialization(self):
        """Test basic builder creation."""
        builder = NeuronBuilder(tau_2=0.05, growing_metric="qe")

        assert builder is not None

    def test_builder_with_mqe_metric(self):
        """Test builder with mean quantization error metric."""
        builder = NeuronBuilder(tau_2=0.05, growing_metric="mqe")

        assert builder is not None


class TestNeuronBuilderZeroNeuron:
    """Tests for zero neuron creation."""

    def test_zero_neuron_creation(self, sample_dataset):
        """Test creation of zero neuron from dataset."""
        builder = NeuronBuilder(tau_2=0.05, growing_metric="qe")

        zero_neuron = builder.zero_neuron(sample_dataset)

        assert zero_neuron is not None
        assert zero_neuron.position == (0, 0)
        assert zero_neuron.has_dataset()

    def test_zero_neuron_weights_are_mean(self, sample_dataset):
        """Test zero neuron weights equal dataset mean."""
        builder = NeuronBuilder(tau_2=0.05, growing_metric="qe")

        zero_neuron = builder.zero_neuron(sample_dataset)
        expected_mean = sample_dataset.mean(axis=0)

        assert np.allclose(zero_neuron.weight_vector(), expected_mean)

    def test_zero_neuron_sets_qe(self, sample_dataset):
        """Test zero neuron sets zero quantization error."""
        builder = NeuronBuilder(tau_2=0.05, growing_metric="qe")

        assert builder.zero_quantization_error is None

        zero_neuron = builder.zero_neuron(sample_dataset)

        assert builder.zero_quantization_error is not None
        assert builder.zero_quantization_error > 0

    def test_zero_neuron_has_dataset(self, sample_dataset):
        """Test zero neuron contains full dataset."""
        builder = NeuronBuilder(tau_2=0.05, growing_metric="qe")

        zero_neuron = builder.zero_neuron(sample_dataset)

        assert np.array_equal(zero_neuron.input_dataset, sample_dataset)


class TestNeuronBuilderNewNeuron:
    """Tests for creating new neurons."""

    def test_new_neuron_creation(self, neuron_builder):
        """Test creating a new neuron."""
        weights_map = [np.random.rand(2, 2, 10)]
        position = (1, 1)

        neuron = neuron_builder.new_neuron(weights_map, position)

        assert neuron is not None
        assert isinstance(neuron, Neuron)
        assert neuron.position == position

    def test_new_neuron_requires_zero_qe(self):
        """Test new_neuron fails without zero_qe set."""
        builder = NeuronBuilder(tau_2=0.05, growing_metric="qe")
        weights_map = [np.random.rand(2, 2, 10)]

        with pytest.raises(AssertionError):
            builder.new_neuron(weights_map, (0, 0))

    def test_new_neuron_with_different_positions(self, neuron_builder):
        """Test creating neurons at different positions."""
        weights_map = [np.random.rand(3, 3, 5)]

        positions = [(0, 0), (0, 1), (1, 0), (2, 2)]
        neurons = [neuron_builder.new_neuron(weights_map, pos) for pos in positions]

        for neuron, pos in zip(neurons, positions):
            assert neuron.position == pos


class TestNeuronBuilderMetrics:
    """Tests for different growing metrics."""

    def test_builder_with_qe_metric(self, sample_dataset):
        """Test builder with quantization error metric."""
        builder = NeuronBuilder(tau_2=0.05, growing_metric="qe")
        zero_neuron = builder.zero_neuron(sample_dataset)

        qe = zero_neuron.compute_quantization_error()

        # QE should be sum of distances
        assert qe > 0

    def test_builder_with_mqe_metric(self, sample_dataset):
        """Test builder with mean quantization error metric."""
        builder = NeuronBuilder(tau_2=0.05, growing_metric="mqe")
        zero_neuron = builder.zero_neuron(sample_dataset)

        mqe = zero_neuron.compute_quantization_error()

        # MQE should be mean of distances (smaller than QE)
        assert mqe > 0

    def test_qe_vs_mqe_difference(self, sample_dataset):
        """Test QE is larger than MQE for same data."""
        builder_qe = NeuronBuilder(tau_2=0.05, growing_metric="qe")
        builder_mqe = NeuronBuilder(tau_2=0.05, growing_metric="mqe")

        zero_neuron_qe = builder_qe.zero_neuron(sample_dataset)
        zero_neuron_mqe = builder_mqe.zero_neuron(sample_dataset)

        qe = zero_neuron_qe.compute_quantization_error()
        mqe = zero_neuron_mqe.compute_quantization_error()

        # QE should be approximately n_samples * MQE
        expected_ratio = sample_dataset.shape[0]
        assert pytest.approx(qe / mqe, rel=0.1) == expected_ratio


class TestNeuronBuilderEdgeCases:
    """Edge case tests for NeuronBuilder."""

    def test_builder_with_tiny_dataset(self, tiny_dataset):
        """Test builder with very small dataset."""
        builder = NeuronBuilder(tau_2=0.05, growing_metric="qe")

        zero_neuron = builder.zero_neuron(tiny_dataset)

        assert zero_neuron is not None
        assert zero_neuron.has_dataset()

    def test_builder_with_single_point(self, single_point_dataset):
        """Test builder with single data point."""
        builder = NeuronBuilder(tau_2=0.05, growing_metric="qe")

        zero_neuron = builder.zero_neuron(single_point_dataset)

        # Zero QE should be 0 for single point matching mean
        assert pytest.approx(builder.zero_quantization_error, abs=1e-6) == 0.0

    def test_builder_with_high_dimensional_data(self, high_dim_dataset):
        """Test builder with high-dimensional data."""
        builder = NeuronBuilder(tau_2=0.05, growing_metric="qe")

        zero_neuron = builder.zero_neuron(high_dim_dataset)

        assert zero_neuron.weight_vector().shape == (1000,)
        assert builder.zero_quantization_error > 0

    def test_builder_with_different_tau2(self, sample_dataset):
        """Test builder with different tau_2 values."""
        tau2_values = [0.01, 0.05, 0.1, 0.5]

        for tau2 in tau2_values:
            builder = NeuronBuilder(tau_2=tau2, growing_metric="qe")
            zero_neuron = builder.zero_neuron(sample_dataset)

            assert zero_neuron is not None
            assert builder.zero_quantization_error > 0
