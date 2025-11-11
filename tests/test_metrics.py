"""Tests for evaluation metrics."""

import numpy as np
import pytest
from ghsom import GHSOM
from ghsom.evaluation.metrics import (
    mean_data_centroid_activation,
    get_total_number_of_neurons,
    dispersion_rate,
    get_number_of_clusters,
    get_number_of_maps,
    get_ghsom_depth,
    get_max_neurons_in_child_map,
    get_child_map_sizes,
    get_mean_child_map_size,
)


class TestMeanDataCentroidActivation:
    """Tests for mean_data_centroid_activation function."""

    def test_mean_activation_basic(self, tiny_dataset):
        """Test basic mean activation calculation."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        mean_act, std_act = mean_data_centroid_activation(root, tiny_dataset)

        # Should return valid numbers
        assert isinstance(mean_act, (float, np.floating))
        assert isinstance(std_act, (float, np.floating))
        assert not np.isnan(mean_act)
        assert not np.isnan(std_act)
        assert mean_act >= 0  # Activation should be non-negative

    def test_mean_activation_with_sample_dataset(self, sample_dataset):
        """Test mean activation with larger dataset."""
        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        mean_act, std_act = mean_data_centroid_activation(root, sample_dataset)

        assert isinstance(mean_act, (float, np.floating))
        assert isinstance(std_act, (float, np.floating))
        assert not np.isnan(mean_act)
        assert not np.isnan(std_act)

    def test_mean_activation_values_reasonable(self, tiny_dataset):
        """Test that activation values are in reasonable range."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        mean_act, std_act = mean_data_centroid_activation(root, tiny_dataset)

        # Activation is Euclidean distance, should be positive
        assert mean_act >= 0
        assert std_act >= 0


class TestGetTotalNumberOfNeurons:
    """Tests for get_total_number_of_neurons function."""

    def test_total_neurons_basic(self, tiny_dataset):
        """Test basic neuron counting."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        total = get_total_number_of_neurons(root)

        # Should have at least 4 neurons (2x2 initial map)
        assert isinstance(total, int)
        assert total >= 4

    def test_total_neurons_matches_hierarchy(self, tiny_dataset):
        """Test neuron count matches actual hierarchy."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        total = get_total_number_of_neurons(root)

        # Should at least have the neurons in root's child map
        assert total >= len(root.child_map.neurons)

    def test_total_neurons_with_larger_hierarchy(self, sample_dataset):
        """Test neuron counting with larger hierarchy."""
        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=2, seed=42, grow_maxiter=5)

        total = get_total_number_of_neurons(root)

        assert isinstance(total, int)
        assert total >= 4


class TestGetNumberOfClusters:
    """Tests for get_number_of_clusters function."""

    def test_number_of_clusters_basic(self, tiny_dataset):
        """Test basic cluster counting."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        num_clusters = get_number_of_clusters(root.child_map)

        # Should return a valid count
        assert isinstance(num_clusters, int)
        assert num_clusters >= 1

    def test_number_of_clusters_with_sample_dataset(self, sample_dataset):
        """Test cluster counting with larger dataset."""
        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        num_clusters = get_number_of_clusters(root.child_map)

        assert isinstance(num_clusters, int)
        assert num_clusters >= 1


class TestDispersionRate:
    """Tests for dispersion_rate function."""

    def test_dispersion_rate_basic(self, tiny_dataset):
        """Test basic dispersion rate calculation."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        dr = dispersion_rate(root, tiny_dataset)

        # Should return a valid number
        assert isinstance(dr, (float, np.floating))
        assert not np.isnan(dr)
        assert dr >= 0  # DR should be non-negative

    def test_dispersion_rate_with_sample_dataset(self, sample_dataset):
        """Test dispersion rate with larger dataset."""
        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        dr = dispersion_rate(root, sample_dataset)

        assert isinstance(dr, (float, np.floating))
        assert not np.isnan(dr)
        assert dr >= 0


class TestGetNumberOfMaps:
    """Tests for get_number_of_maps function."""

    def test_number_of_maps_basic(self, tiny_dataset):
        """Test basic map counting."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        num_maps = get_number_of_maps(root)

        assert isinstance(num_maps, int)
        assert num_maps >= 1


class TestGetGHSOMDepth:
    """Tests for get_ghsom_depth function."""

    def test_ghsom_depth_basic(self, tiny_dataset):
        """Test basic depth calculation."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        depth = get_ghsom_depth(root.child_map, level=0)

        assert isinstance(depth, int)
        assert depth >= 0


class TestGetMaxNeuronsInChildMap:
    """Tests for get_max_neurons_in_child_map function."""

    def test_max_neurons_basic(self, tiny_dataset):
        """Test basic max neurons calculation."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        max_neurons = get_max_neurons_in_child_map(root.child_map)

        assert isinstance(max_neurons, int)
        assert max_neurons >= 4  # At least 2x2


class TestGetChildMapSizes:
    """Tests for get_child_map_sizes function."""

    def test_child_map_sizes_basic(self, tiny_dataset):
        """Test child map sizes collection."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        sizes = get_child_map_sizes(root.child_map)

        assert isinstance(sizes, list)
        assert len(sizes) > 0


class TestGetMeanChildMapSize:
    """Tests for get_mean_child_map_size function."""

    def test_mean_child_map_size_basic(self, tiny_dataset):
        """Test mean child map size calculation."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        mean_size = get_mean_child_map_size(root.child_map)

        # Returns array or scalar
        assert mean_size is not None
        if isinstance(mean_size, np.ndarray):
            assert len(mean_size) > 0
        else:
            assert mean_size >= 0
