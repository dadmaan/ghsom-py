"""Tests for I/O persistence functionality."""

import pytest
import tempfile
import pickle
from pathlib import Path
import numpy as np


class TestIOPersistence:
    """Tests for save/load functionality."""

    def test_save_load_imports_available(self):
        """Test that save/load functions can be imported."""
        from ghsom.io import save_model, load_model

        assert save_model is not None
        assert load_model is not None

    def test_save_model_creates_file(self, sample_dataset):
        """Test save_model creates a file."""
        from ghsom import GHSOM
        from ghsom.io import save_model

        # Create a simple GHSOM model
        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            filepath = f.name

        try:
            save_model(ghsom, filepath)

            # Check file was created
            assert Path(filepath).exists()
            assert Path(filepath).stat().st_size > 0
        finally:
            # Cleanup
            if Path(filepath).exists():
                Path(filepath).unlink()

    def test_load_model_restores_object(self, sample_dataset):
        """Test load_model can restore a saved model."""
        from ghsom import GHSOM
        from ghsom.io import save_model, load_model

        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            filepath = f.name

        try:
            save_model(ghsom, filepath)
            loaded_model = load_model(filepath)

            assert loaded_model is not None
            assert isinstance(loaded_model, GHSOM)
        finally:
            if Path(filepath).exists():
                Path(filepath).unlink()

    def test_save_load_roundtrip_preserves_data(self, sample_dataset):
        """Test save/load roundtrip preserves model data."""
        from ghsom import GHSOM
        from ghsom.io import save_model, load_model

        original_ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            filepath = f.name

        try:
            save_model(original_ghsom, filepath)
            loaded_ghsom = load_model(filepath)

            # Compare attributes (basic check)
            assert hasattr(loaded_ghsom, "_GHSOM__t1")
            assert hasattr(loaded_ghsom, "_GHSOM__t2")
        finally:
            if Path(filepath).exists():
                Path(filepath).unlink()


class TestIOParsing:
    """Tests for parsing functionality."""

    def test_parsing_module_imports(self):
        """Test parsing module can be imported."""
        from ghsom.io import parsing

        assert parsing is not None


class TestIOEdgeCases:
    """Edge case tests for I/O operations."""

    def test_load_nonexistent_file_raises_error(self):
        """Test loading nonexistent file raises error."""
        from ghsom.io import load_model

        with pytest.raises((FileNotFoundError, OSError)):
            load_model("/tmp/nonexistent_model_12345.pkl")

    def test_save_to_invalid_path_raises_error(self, sample_dataset):
        """Test saving to invalid path raises error."""
        from ghsom import GHSOM
        from ghsom.io import save_model

        ghsom = GHSOM(
            input_dataset=sample_dataset,
            t1=0.5,
            t2=0.05,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0,
        )

        with pytest.raises((OSError, PermissionError, FileNotFoundError)):
            save_model(ghsom, "/invalid/path/that/does/not/exist/model.pkl")
