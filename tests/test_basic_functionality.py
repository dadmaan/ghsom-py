"""Test basic functionality of GHSOM package."""
import numpy as np
import pytest


def test_ghsom_instantiation():
    """Test that GHSOM can be instantiated."""
    from ghsom import GHSOM

    data = np.random.rand(50, 10)
    ghsom = GHSOM(
        input_dataset=data,
        t1=0.5,
        t2=0.05,
        learning_rate=0.1,
        decay=0.9,
        gaussian_sigma=1.0,
    )

    assert ghsom is not None


def test_gsom_instantiation():
    """Test that GSOM can be instantiated."""
    from ghsom.core import GSOM
    from ghsom import NeuronBuilder

    # GSOM requires many parameters for initialization
    # This is typically done internally by GHSOM, so we just test basic instantiation
    data = np.random.rand(30, 5)
    weights_map = np.random.rand(2, 2, 5)
    builder = NeuronBuilder(tau_2=0.05, growing_metric="qe")
    builder.zero_quantization_error = 0.1

    gsom = GSOM(
        initial_map_size=(2, 2),
        parent_quantization_error=0.1,
        t1=0.5,
        data_size=30,
        weights_map=weights_map,
        parent_dataset=data,
        neuron_builder=builder,
    )

    assert gsom is not None


def test_neuron_class_exists():
    """Test that Neuron class can be imported."""
    from ghsom.core import Neuron

    # Neuron requires complex weight_map structure
    # It's typically instantiated by NeuronBuilder in the GHSOM algorithm
    # Just verify the class exists
    assert Neuron is not None


def test_neuron_builder():
    """Test that NeuronBuilder can be instantiated."""
    from ghsom import NeuronBuilder

    builder = NeuronBuilder(tau_2=0.05, growing_metric="qe")
    assert builder is not None


def test_tracking_callback_is_abstract():
    """Test that TrackingCallback cannot be instantiated directly."""
    from ghsom import TrackingCallback

    with pytest.raises(TypeError):
        TrackingCallback()


def test_save_load_model(tmp_path):
    """Test that models can be saved and loaded."""
    from ghsom.io import save_model, load_model

    # Create a simple object to save
    test_data = {"key": "value", "array": np.array([1, 2, 3])}

    # Save to temporary file
    filepath = tmp_path / "test_model.pkl"
    save_model(test_data, str(filepath))

    # Load back
    loaded_data = load_model(str(filepath))

    assert loaded_data["key"] == "value"
    assert np.array_equal(loaded_data["array"], np.array([1, 2, 3]))
