"""Test that all package imports work correctly."""
import pytest


def test_core_imports():
    """Test that core classes can be imported."""
    from ghsom import GHSOM, GSOM, Neuron

    assert GHSOM is not None
    assert GSOM is not None
    assert Neuron is not None


def test_builder_imports():
    """Test that builder classes can be imported."""
    from ghsom import NeuronBuilder

    assert NeuronBuilder is not None


def test_callback_imports():
    """Test that callback classes can be imported."""
    from ghsom import TrackingCallback
    from ghsom.callbacks import WandBCallback

    assert TrackingCallback is not None
    assert WandBCallback is not None


def test_io_imports():
    """Test that I/O functions can be imported."""
    from ghsom.io import save_model, load_model

    assert save_model is not None
    assert load_model is not None


def test_version():
    """Test that version is accessible."""
    from ghsom import __version__

    assert __version__ is not None
    assert isinstance(__version__, str)
    assert __version__ == "0.1.0"


def test_all_exports():
    """Test that __all__ contains expected exports."""
    import ghsom

    expected_exports = [
        "GHSOM",
        "GSOM",
        "Neuron",
        "NeuronBuilder",
        "TrackingCallback",
        "__version__",
    ]

    for export in expected_exports:
        assert export in ghsom.__all__
        assert hasattr(ghsom, export)
