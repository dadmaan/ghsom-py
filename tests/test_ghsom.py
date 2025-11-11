"""Tests for GHSOM class."""

import numpy as np
import pytest
from ghsom import GHSOM


class TestGHSOMInitialization:
    """Tests for GHSOM initialization."""

    def test_ghsom_basic_initialization(self, sample_dataset, ghsom_default_config):
        """Test basic GHSOM creation."""
        ghsom = GHSOM(input_dataset=sample_dataset, **ghsom_default_config)

        assert ghsom is not None
        assert hasattr(ghsom, "train")

    def test_ghsom_with_different_t1_values(self, sample_dataset):
        """Test GHSOM with various t1 values."""
        t1_values = [0.1, 0.3, 0.5, 0.7, 0.9]

        for t1 in t1_values:
            ghsom = GHSOM(
                input_dataset=sample_dataset,
                t1=t1,
                t2=0.05,
                learning_rate=0.1,
                decay=0.9,
                gaussian_sigma=1.0,
            )
            assert ghsom is not None

    def test_ghsom_with_different_t2_values(self, sample_dataset):
        """Test GHSOM with various t2 values."""
        t2_values = [0.01, 0.05, 0.1, 0.2]

        for t2 in t2_values:
            ghsom = GHSOM(
                input_dataset=sample_dataset,
                t1=0.5,
                t2=t2,
                learning_rate=0.1,
                decay=0.9,
                gaussian_sigma=1.0,
            )
            assert ghsom is not None

    def test_ghsom_with_logger(self, sample_dataset):
        """Test GHSOM with custom logger."""
        import logging

        custom_logger = logging.getLogger("test_ghsom")

        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
            logger=custom_logger,
        )

        assert ghsom.logger is custom_logger

    def test_ghsom_with_mqe_metric(self, sample_dataset):
        """Test GHSOM with mean quantization error metric."""
        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
            growing_metric="mqe",
        )

        assert ghsom is not None


class TestGHSOMParameters:
    """Tests for GHSOM parameter handling."""

    def test_ghsom_stores_parameters(self, sample_dataset):
        """Test GHSOM stores initialization parameters."""
        t1, t2 = 0.6, 0.03
        lr, decay, sigma = 0.15, 0.85, 1.5

        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=t1,
            t2=t2,
            learning_rate=lr,
            decay=decay,
            gaussian_sigma=sigma,
        )

        # Parameters stored as private attributes
        assert hasattr(ghsom, "_GHSOM__t1")
        assert hasattr(ghsom, "_GHSOM__t2")

    def test_ghsom_with_shallow_config(self, sample_dataset, ghsom_shallow_config):
        """Test GHSOM with shallow hierarchy config."""
        ghsom = GHSOM(input_dataset=sample_dataset, **ghsom_shallow_config)

        assert ghsom is not None

    def test_ghsom_with_deep_config(self, sample_dataset, ghsom_deep_config):
        """Test GHSOM with deep hierarchy config."""
        ghsom = GHSOM(input_dataset=sample_dataset, **ghsom_deep_config)

        assert ghsom is not None


class TestGHSOMDataset:
    """Tests for GHSOM dataset handling."""

    def test_ghsom_with_tiny_dataset(self, tiny_dataset):
        """Test GHSOM with tiny dataset."""
        ghsom = GHSOM(
            input_dataset=tiny_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        assert ghsom is not None

    def test_ghsom_with_large_dataset(self, large_dataset):
        """Test GHSOM with large dataset."""
        ghsom = GHSOM(
            input_dataset=large_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        assert ghsom is not None

    def test_ghsom_with_high_dimensional_data(self, high_dim_dataset):
        """Test GHSOM with high-dimensional dataset."""
        ghsom = GHSOM(
            input_dataset=high_dim_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        assert ghsom is not None

    def test_ghsom_stores_input_dimension(self, sample_dataset):
        """Test GHSOM stores input dimension correctly."""
        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        assert hasattr(ghsom, "_GHSOM__input_dimension")


class TestGHSOMEdgeCases:
    """Edge case tests for GHSOM."""

    def test_ghsom_with_zero_learning_rate(self, sample_dataset):
        """Test GHSOM with zero learning rate (edge case)."""
        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.0,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        # Should initialize without error
        assert ghsom is not None

    def test_ghsom_with_extreme_t1_t2(self, sample_dataset):
        """Test GHSOM with extreme t1/t2 combinations."""
        # Very high t1, very low t2
        ghsom1 = GHSOM(
            input_dataset=sample_dataset,
            t1=0.99,
            t2=0.001,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )
        assert ghsom1 is not None

        # Very low t1, very high t2
        ghsom2 = GHSOM(
            input_dataset=sample_dataset,
            t1=0.01,
            t2=0.5,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )
        assert ghsom2 is not None

    def test_ghsom_with_high_decay(self, sample_dataset):
        """Test GHSOM with high decay value."""
        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.99,
            gaussian_sigma=1.0,
        )

        assert ghsom is not None

    def test_ghsom_with_low_decay(self, sample_dataset):
        """Test GHSOM with low decay value."""
        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.1,
            gaussian_sigma=1.0,
        )

        assert ghsom is not None
