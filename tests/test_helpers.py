"""Tests for helper functions."""

import numpy as np
import pytest
from ghsom import GHSOM
from ghsom.utils.helpers import get_used_neurons, get_random_sample_used_neurons


class TestGetUsedNeurons:
    """Tests for get_used_neurons function."""

    def test_get_used_neurons_basic(self, tiny_dataset):
        """Test basic used neurons retrieval."""
        # Train a simple GHSOM
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        # Create labels for the dataset
        labels = ["A", "B"] * (len(tiny_dataset) // 2)
        if len(labels) < len(tiny_dataset):
            labels.append("A")

        # Get used neurons
        used_neurons = get_used_neurons(root, tiny_dataset, labels)

        # Should return a dictionary
        assert isinstance(used_neurons, dict)
        assert len(used_neurons) > 0

    def test_get_used_neurons_with_single_label(self, tiny_dataset):
        """Test get_used_neurons with single label."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        # All same label
        labels = ["A"] * len(tiny_dataset)

        used_neurons = get_used_neurons(root, tiny_dataset, labels)

        # Should have neurons with label "A"
        assert isinstance(used_neurons, dict)
        for neuron, label_dict in used_neurons.items():
            assert "A" in label_dict

    def test_get_used_neurons_with_multiple_labels(self, sample_dataset):
        """Test get_used_neurons with multiple labels."""
        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        # Multiple labels
        labels = ["A", "B", "C", "D"] * (len(sample_dataset) // 4 + 1)
        labels = labels[: len(sample_dataset)]

        used_neurons = get_used_neurons(root, sample_dataset, labels)

        assert isinstance(used_neurons, dict)
        assert len(used_neurons) > 0

    def test_get_used_neurons_counts_correctly(self, tiny_dataset):
        """Test that get_used_neurons counts labels correctly."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        labels = ["A"] * len(tiny_dataset)

        used_neurons = get_used_neurons(root, tiny_dataset, labels)

        # Total count should equal dataset size
        total_count = sum(
            sum(label_counts.values()) for label_counts in used_neurons.values()
        )
        assert total_count == len(tiny_dataset)


class TestGetRandomSampleUsedNeurons:
    """Tests for get_random_sample_used_neurons function."""

    def test_get_random_sample_basic(self, tiny_dataset):
        """Test basic random sampling of used neurons."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        labels = ["A", "B"] * (len(tiny_dataset) // 2)
        if len(labels) < len(tiny_dataset):
            labels.append("A")

        sample_neurons, genre_tags = get_random_sample_used_neurons(
            root, tiny_dataset, labels, sample_size=1
        )

        # Should return expected structure
        assert isinstance(sample_neurons, list)
        assert isinstance(genre_tags, list)
        assert len(sample_neurons) == 1
        assert len(genre_tags) == 1

    def test_get_random_sample_multiple_neurons(self, sample_dataset):
        """Test sampling multiple neurons."""
        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        labels = ["A", "B", "C"] * (len(sample_dataset) // 3 + 1)
        labels = labels[: len(sample_dataset)]

        # Sample 2 neurons
        sample_neurons, genre_tags = get_random_sample_used_neurons(
            root, sample_dataset, labels, sample_size=2
        )

        assert len(sample_neurons) == 2
        assert len(genre_tags) == 2

    def test_get_random_sample_with_default_size(self, tiny_dataset):
        """Test random sampling with default sample size."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        root = ghsom.train(epochs_number=2, seed=42, grow_maxiter=3)

        labels = ["A"] * len(tiny_dataset)

        # Default sample_size=1
        sample_neurons, genre_tags = get_random_sample_used_neurons(
            root, tiny_dataset, labels
        )

        assert len(sample_neurons) == 1
        assert len(genre_tags) == 1
