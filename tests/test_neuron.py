"""Comprehensive tests for Neuron class."""

import numpy as np
import pytest
from ghsom.core import Neuron
from ghsom.builders import NeuronBuilder


class TestNeuronInitialization:
    """Tests for Neuron initialization."""

    def test_neuron_basic_initialization(self, neuron_builder):
        """Test basic neuron creation via NeuronBuilder."""
        weights_map = [np.random.rand(2, 2, 10)]
        position = (0, 1)

        neuron = neuron_builder.new_neuron(weights_map, position)

        assert neuron is not None
        assert neuron.position == position
        assert neuron.weight_vector().shape == (10,)

    def test_neuron_initializes_with_correct_dimensions(self, neuron_builder):
        """Test neuron weight dimensions are preserved."""
        dim = 20
        weights_map = [np.random.rand(2, 2, dim)]
        neuron = neuron_builder.new_neuron(weights_map, (1, 1))

        assert neuron.weight_vector().shape == (dim,)

    def test_neuron_coordinates_stored_correctly(self, neuron_builder):
        """Test neuron coordinates are accessible."""
        position = (5, 7)
        weights_map = [np.random.rand(10, 10, 3)]
        neuron = neuron_builder.new_neuron(weights_map, position)

        assert neuron.position == position


class TestNeuronActivation:
    """Tests for neuron activation calculation."""

    def test_activation_with_identical_input(self, neuron_builder):
        """Test activation when input matches weights exactly."""
        weights_map = [np.array([[[0.5, 0.5, 0.5]]])]  # 1x1x3
        neuron = neuron_builder.new_neuron(weights_map, (0, 0))

        activation = neuron.activation(np.array([0.5, 0.5, 0.5]))

        # Activation should be 0 for identical vectors
        assert pytest.approx(activation, abs=1e-6) == 0.0

    def test_activation_euclidean_distance(self, neuron_builder):
        """Test activation computes Euclidean distance."""
        weights_map = [np.array([[[0.0, 0.0, 0.0]]])]
        neuron = neuron_builder.new_neuron(weights_map, (0, 0))

        input_vector = np.array([3.0, 4.0, 0.0])
        activation = neuron.activation(input_vector)

        # Distance should be 5 (3-4-5 triangle)
        assert pytest.approx(activation, abs=1e-6) == 5.0

    def test_activation_with_different_inputs(self, neuron_builder):
        """Test activation changes with different inputs."""
        weights_map = [np.array([[[0.5, 0.5]]])]
        neuron = neuron_builder.new_neuron(weights_map, (0, 0))

        activation1 = neuron.activation(np.array([0.5, 0.5]))
        activation2 = neuron.activation(np.array([1.0, 1.0]))

        assert activation1 < activation2

    def test_activation_with_batch_input(self, neuron_builder):
        """Test activation with multiple inputs at once."""
        weights_map = [np.array([[[0.0, 0.0]]])]
        neuron = neuron_builder.new_neuron(weights_map, (0, 0))

        batch_input = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        activations = neuron.activation(batch_input)

        assert activations.shape == (3,)
        assert pytest.approx(activations[0], abs=1e-6) == 1.0
        assert pytest.approx(activations[1], abs=1e-6) == 1.0
        assert pytest.approx(activations[2], abs=1e-6) == np.sqrt(2)


class TestNeuronQuantizationError:
    """Tests for quantization error calculations."""

    def test_quantization_error_single_point(self, neuron_builder, sample_dataset):
        """Test QE calculation with single data point."""
        weights_map = [np.zeros((1, 1, 10))]
        neuron = neuron_builder.new_neuron(weights_map, (0, 0))
        data = sample_dataset[0:1]

        neuron.replace_dataset(data)
        qe = neuron.compute_quantization_error()

        assert qe >= 0
        assert isinstance(qe, (int, float, np.number))

    def test_quantization_error_with_dataset(self, neuron_builder, sample_dataset):
        """Test QE with full dataset."""
        weights_map = [np.random.rand(1, 1, 10)]
        neuron = neuron_builder.new_neuron(weights_map, (0, 0))
        neuron.replace_dataset(sample_dataset)

        qe = neuron.compute_quantization_error()

        assert qe > 0
        assert np.isfinite(qe)

    def test_quantization_error_zero_for_identical_data(self, neuron_builder):
        """Test QE is zero when neuron perfectly represents data."""
        weights_map = [np.array([[[0.5, 0.5, 0.5]]])]
        neuron = neuron_builder.new_neuron(weights_map, (0, 0))

        # Dataset with identical points to weights
        neuron.replace_dataset(np.array([[0.5, 0.5, 0.5]]))

        qe = neuron.compute_quantization_error()
        assert pytest.approx(qe, abs=1e-6) == 0.0


class TestNeuronDataset:
    """Tests for neuron dataset management."""

    def test_dataset_assignment(self, neuron_builder, sample_dataset):
        """Test dataset can be assigned to neuron."""
        weights_map = [np.random.rand(1, 1, 10)]
        neuron = neuron_builder.new_neuron(weights_map, (0, 0))

        neuron.replace_dataset(sample_dataset)

        assert neuron.input_dataset is not None
        assert neuron.input_dataset.shape == sample_dataset.shape
        assert neuron.has_dataset()

    def test_dataset_is_empty_initially(self, neuron_builder):
        """Test neuron has empty dataset initially."""
        weights_map = [np.random.rand(1, 1, 3)]
        neuron = neuron_builder.new_neuron(weights_map, (0, 0))

        assert not neuron.has_dataset()
        assert neuron.input_dataset.shape[0] == 0

    def test_dataset_clear(self, neuron_builder, sample_dataset):
        """Test dataset can be cleared."""
        weights_map = [np.random.rand(1, 1, 10)]
        neuron = neuron_builder.new_neuron(weights_map, (0, 0))

        neuron.replace_dataset(sample_dataset)
        assert neuron.has_dataset()

        neuron.clear_dataset()
        assert not neuron.has_dataset()


class TestNeuronChildMap:
    """Tests for neuron child map functionality."""

    def test_neuron_has_child_map_attribute(self, neuron_builder):
        """Test neuron can have a child map."""
        weights_map = [np.random.rand(1, 1, 3)]
        neuron = neuron_builder.new_neuron(weights_map, (0, 0))

        assert hasattr(neuron, "child_map")

    def test_neuron_child_map_initially_none(self, neuron_builder):
        """Test child map is None by default."""
        weights_map = [np.random.rand(1, 1, 3)]
        neuron = neuron_builder.new_neuron(weights_map, (0, 0))

        assert neuron.child_map is None

    def test_needs_child_map(self, neuron_builder, sample_dataset):
        """Test needs_child_map logic."""
        weights_map = [np.random.rand(1, 1, 10)]
        neuron = neuron_builder.new_neuron(weights_map, (0, 0))
        neuron.replace_dataset(sample_dataset)

        # Result depends on QE vs threshold
        result = neuron.needs_child_map()
        assert isinstance(result, (bool, np.bool_))


class TestNeuronWeightDistance:
    """Tests for weight distance calculations."""

    def test_distance_to_identical_weights(self, neuron_builder):
        """Test distance to identical weights is zero."""
        weights = np.array([[[1.0, 2.0, 3.0]]])
        neuron1 = neuron_builder.new_neuron([weights], (0, 0))
        neuron2 = neuron_builder.new_neuron([weights.copy()], (0, 0))

        distance = neuron1.weight_distance_from_other_unit(neuron2)

        assert pytest.approx(distance, abs=1e-6) == 0.0

    def test_distance_to_different_weights(self, neuron_builder):
        """Test distance to different weights is positive."""
        weights1 = np.array([[[0.0, 0.0]]])
        weights2 = np.array([[[1.0, 1.0]]])

        neuron1 = neuron_builder.new_neuron([weights1], (0, 0))
        neuron2 = neuron_builder.new_neuron([weights2], (0, 0))

        distance = neuron1.weight_distance_from_other_unit(neuron2)

        assert distance > 0
        assert pytest.approx(distance, abs=1e-6) == np.sqrt(2)


class TestNeuronEdgeCases:
    """Edge case tests for Neuron."""

    def test_neuron_with_single_dimension(self, neuron_builder):
        """Test neuron with 1D weights."""
        weights_map = [np.array([[[0.5]]])]
        neuron = neuron_builder.new_neuron(weights_map, (0, 0))

        activation = neuron.activation(np.array([0.7]))

        assert pytest.approx(activation, abs=1e-6) == 0.2

    def test_neuron_with_high_dimensional_weights(self, neuron_builder):
        """Test neuron with high-dimensional weights."""
        dim = 1000
        weights_map = [np.random.rand(1, 1, dim)]
        neuron = neuron_builder.new_neuron(weights_map, (0, 0))

        assert neuron.weight_vector().shape == (dim,)

        input_vec = np.random.rand(dim)
        activation = neuron.activation(input_vec)

        assert np.isfinite(activation)
        assert activation >= 0

    def test_neuron_with_zero_weights(self, neuron_builder):
        """Test neuron initialized with zero weights."""
        weights_map = [np.zeros((1, 1, 10))]
        neuron = neuron_builder.new_neuron(weights_map, (0, 0))

        assert np.allclose(neuron.weight_vector(), 0)

        activation = neuron.activation(np.ones(10))
        expected = np.sqrt(10)  # Distance from origin to (1,1,...,1)

        assert pytest.approx(activation, abs=1e-6) == expected
