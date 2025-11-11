"""Tests for callback functionality."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from ghsom.callbacks.base import TrackingCallback


class MockCallback(TrackingCallback):
    """Mock callback for testing."""

    def __init__(self):
        self.train_begin_called = False
        self.map_created_called = False
        self.train_end_called = False
        self.config = None
        self.metrics_log = []
        self.results = None

    def on_train_begin(self, config):
        self.train_begin_called = True
        self.config = config

    def on_map_created(self, metrics):
        self.map_created_called = True
        self.metrics_log.append(metrics)

    def on_train_end(self, results):
        self.train_end_called = True
        self.results = results


class TestTrackingCallback:
    """Tests for base TrackingCallback class."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that TrackingCallback cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TrackingCallback()

    def test_mock_callback_implements_interface(self):
        """Test that MockCallback properly implements TrackingCallback."""
        callback = MockCallback()
        assert isinstance(callback, TrackingCallback)
        assert hasattr(callback, "on_train_begin")
        assert hasattr(callback, "on_map_created")
        assert hasattr(callback, "on_train_end")

    def test_callback_methods_callable(self):
        """Test callback methods are callable."""
        callback = MockCallback()

        assert callable(callback.on_train_begin)
        assert callable(callback.on_map_created)
        assert callable(callback.on_train_end)


class TestCallbackExecution:
    """Tests for callback execution."""

    def test_on_train_begin_records_config(self):
        """Test on_train_begin receives and stores config."""
        callback = MockCallback()
        config = {"t1": 0.5, "t2": 0.05, "epochs": 10}

        callback.on_train_begin(config)

        assert callback.train_begin_called
        assert callback.config == config

    def test_on_map_created_records_metrics(self):
        """Test on_map_created receives and stores metrics."""
        callback = MockCallback()
        metrics = {"layer": 0, "neurons": 4, "qe": 0.5}

        callback.on_map_created(metrics)

        assert callback.map_created_called
        assert len(callback.metrics_log) == 1
        assert callback.metrics_log[0] == metrics

    def test_on_train_end_records_results(self):
        """Test on_train_end receives and stores results."""
        callback = MockCallback()
        results = {"total_layers": 3, "total_neurons": 15}

        callback.on_train_end(results)

        assert callback.train_end_called
        assert callback.results == results

    def test_multiple_map_created_calls(self):
        """Test multiple on_map_created calls are recorded."""
        callback = MockCallback()

        for i in range(5):
            callback.on_map_created({"layer": i, "qe": 0.1 * i})

        assert len(callback.metrics_log) == 5
        assert callback.metrics_log[0]["layer"] == 0
        assert callback.metrics_log[4]["layer"] == 4


class TestMultipleCallbacks:
    """Tests for using multiple callbacks."""

    def test_multiple_callbacks_instantiation(self):
        """Test creating multiple callback instances."""
        callback1 = MockCallback()
        callback2 = MockCallback()

        assert callback1 is not callback2
        assert not callback1.train_begin_called
        assert not callback2.train_begin_called

    def test_multiple_callbacks_independent(self):
        """Test callbacks are independent."""
        callback1 = MockCallback()
        callback2 = MockCallback()

        callback1.on_train_begin({"epochs": 10})

        assert callback1.train_begin_called
        assert not callback2.train_begin_called

    def test_callbacks_list_execution(self):
        """Test executing callbacks from a list."""
        callbacks = [MockCallback() for _ in range(3)]
        config = {"t1": 0.5}

        for cb in callbacks:
            cb.on_train_begin(config)

        assert all(cb.train_begin_called for cb in callbacks)
        assert all(cb.config == config for cb in callbacks)


class TestCallbackErrorHandling:
    """Tests for callback error handling."""

    def test_callback_with_none_values(self):
        """Test callback handles None values gracefully."""
        callback = MockCallback()

        callback.on_train_begin(None)
        callback.on_map_created(None)
        callback.on_train_end(None)

        assert callback.train_begin_called
        assert callback.map_created_called
        assert callback.train_end_called

    def test_callback_with_empty_dict(self):
        """Test callback handles empty dictionaries."""
        callback = MockCallback()

        callback.on_train_begin({})
        callback.on_map_created({})
        callback.on_train_end({})

        assert callback.config == {}
        assert callback.metrics_log == [{}]
        assert callback.results == {}


class TestWandBCallback:
    """Tests for WandB callback (mocked)."""

    def test_wandb_callback_can_be_imported(self):
        """Test WandB callback can be imported."""
        from ghsom.callbacks import WandBCallback

        assert WandBCallback is not None


class TestCustomCallback:
    """Tests for custom callback implementations."""

    def test_custom_callback_minimal_implementation(self):
        """Test minimal custom callback implementation."""

        class MinimalCallback(TrackingCallback):
            def on_train_begin(self, config):
                pass

            def on_map_created(self, metrics):
                pass

            def on_train_end(self, results):
                pass

        callback = MinimalCallback()

        # Should not raise any errors
        callback.on_train_begin({})
        callback.on_map_created({})
        callback.on_train_end({})

    def test_custom_callback_with_logging(self):
        """Test custom callback that logs to a list."""

        class LoggingCallback(TrackingCallback):
            def __init__(self):
                self.log = []

            def on_train_begin(self, config):
                self.log.append(("begin", config))

            def on_map_created(self, metrics):
                self.log.append(("map", metrics))

            def on_train_end(self, results):
                self.log.append(("end", results))

        callback = LoggingCallback()

        callback.on_train_begin({"epochs": 5})
        callback.on_map_created({"layer": 0})
        callback.on_map_created({"layer": 1})
        callback.on_train_end({"status": "complete"})

        assert len(callback.log) == 4
        assert callback.log[0][0] == "begin"
        assert callback.log[1][0] == "map"
        assert callback.log[3][0] == "end"
