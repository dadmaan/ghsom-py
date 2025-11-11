# Tutorial 2: Using Callbacks for Training Tracking

Learn how to monitor and track your GHSOM training process using callbacks.

## What You'll Learn

- How to create custom callbacks
- How to use built-in WandB callback
- How to track training metrics
- How to implement custom logging

## What are Callbacks?

Callbacks are hooks that let you execute custom code at specific points during training:

- `on_train_begin()`: Called when training starts
- `on_map_created()`: Called when a new map is created in the hierarchy
- `on_train_end()`: Called when training completes

## Creating a Simple Callback

```python
import numpy as np
from ghsom import GHSOM
from ghsom.callbacks import TrackingCallback

class SimpleCallback(TrackingCallback):
    def on_train_begin(self, config):
        print("=" * 50)
        print("Training Started")
        print("=" * 50)
        print(f"Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print()

    def on_map_created(self, metrics):
        qe = metrics.get('qe', metrics.get('mqe', 'N/A'))
        position = metrics.get('neuron_position', 'unknown')
        print(f"✓ New map created at position {position} (QE: {qe:.4f})")

    def on_train_end(self, results):
        print()
        print("=" * 50)
        print("Training Complete")
        print("=" * 50)
        print(f"Total neurons created: {results['total_neurons']}")

# Use the callback
data = np.random.rand(100, 10)
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
model = ghsom.train(epochs_number=30, callbacks=[SimpleCallback()])
```

## Tracking Metrics with a Callback

```python
from ghsom.callbacks import TrackingCallback
import time

class MetricsCallback(TrackingCallback):
    def __init__(self):
        self.start_time = None
        self.maps_created = 0
        self.qe_values = []

    def on_train_begin(self, config):
        self.start_time = time.time()
        self.maps_created = 0
        self.qe_values = []
        print(f"Starting training with t1={config['t1']}, t2={config['t2']}")

    def on_map_created(self, metrics):
        self.maps_created += 1
        qe = metrics.get('qe', metrics.get('mqe'))
        if qe is not None:
            self.qe_values.append(qe)

        if self.maps_created % 5 == 0:
            avg_qe = np.mean(self.qe_values[-5:]) if self.qe_values else 0
            print(f"Maps created: {self.maps_created}, Avg QE (last 5): {avg_qe:.4f}")

    def on_train_end(self, results):
        elapsed = time.time() - self.start_time
        print(f"\nTraining completed in {elapsed:.2f} seconds")
        print(f"Total maps created: {self.maps_created}")
        if self.qe_values:
            print(f"Average QE: {np.mean(self.qe_values):.4f}")
            print(f"Final QE: {self.qe_values[-1]:.4f}")

# Use the metrics callback
data = np.random.rand(150, 8)
ghsom = GHSOM(input_dataset=data, t1=0.4, t2=0.05)
model = ghsom.train(epochs_number=40, callbacks=[MetricsCallback()])
```

## Logging to a File

```python
from ghsom.callbacks import TrackingCallback
import json
from datetime import datetime

class FileLoggingCallback(TrackingCallback):
    def __init__(self, filename="training_log.json"):
        self.filename = filename
        self.log = {
            "started_at": None,
            "config": None,
            "maps": [],
            "completed_at": None
        }

    def on_train_begin(self, config):
        self.log["started_at"] = datetime.now().isoformat()
        self.log["config"] = config

    def on_map_created(self, metrics):
        self.log["maps"].append({
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        })

    def on_train_end(self, results):
        self.log["completed_at"] = datetime.now().isoformat()
        self.log["results"] = results

        # Save to file
        with open(self.filename, 'w') as f:
            json.dump(self.log, f, indent=2)

        print(f"Training log saved to {self.filename}")

# Use file logging
data = np.random.rand(100, 10)
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
model = ghsom.train(
    epochs_number=30,
    callbacks=[FileLoggingCallback("my_training.json")]
)
```

## Using Multiple Callbacks

You can use multiple callbacks simultaneously:

```python
# Combine callbacks
callbacks = [
    SimpleCallback(),
    MetricsCallback(),
    FileLoggingCallback("training_log.json")
]

data = np.random.rand(100, 10)
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
model = ghsom.train(epochs_number=30, callbacks=callbacks)
```

## WandB Integration

For experiment tracking with Weights & Biases:

```python
from ghsom import GHSOM
from ghsom.callbacks import WandBCallback
import numpy as np

# First install: pip install ghsom-py[wandb]

# Initialize WandB callback
wandb_callback = WandBCallback(
    project="ghsom-experiments",
    name="experiment-001",
    config={
        "t1": 0.5,
        "t2": 0.05,
        "learning_rate": 0.1,
        "decay": 0.9,
        "gaussian_sigma": 1.0
    },
    tags=["tutorial", "baseline"]
)

# Train with WandB tracking
data = np.random.rand(200, 12)
ghsom = GHSOM(
    input_dataset=data,
    t1=0.5,
    t2=0.05,
    learning_rate=0.1,
    decay=0.9,
    gaussian_sigma=1.0
)

model = ghsom.train(
    epochs_number=50,
    callbacks=[wandb_callback]
)
```

The WandB callback automatically logs:
- Training configuration
- Maps created over time
- Quantization error metrics
- Total neurons in the final model

## Advanced: Custom Validation Callback

```python
from ghsom.callbacks import TrackingCallback
from ghsom.evaluation.metrics import mean_data_centroid_activation

class ValidationCallback(TrackingCallback):
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.model = None

    def on_train_begin(self, config):
        print("Validation data ready:", self.validation_data.shape)

    def on_map_created(self, metrics):
        pass  # Could validate after each map creation

    def on_train_end(self, results):
        # Note: In practice, you'd need access to the trained model
        # This is a conceptual example
        print("\nRunning validation...")
        print(f"Validation dataset size: {len(self.validation_data)}")

# Usage
train_data = np.random.rand(150, 10)
val_data = np.random.rand(50, 10)

ghsom = GHSOM(input_dataset=train_data, t1=0.5, t2=0.05)
model = ghsom.train(
    epochs_number=30,
    callbacks=[ValidationCallback(val_data)]
)
```

## Best Practices

1. **Keep callbacks lightweight**: Avoid expensive operations in callbacks
2. **Use appropriate logging levels**: Don't log too frequently in `on_map_created`
3. **Handle errors gracefully**: Add try-except blocks in callback methods
4. **Close resources**: Clean up file handles or connections in `on_train_end`
5. **Combine callbacks**: Use multiple callbacks for different purposes

## Next Steps

- [Tutorial 3: Model Persistence](03_persistence.md) - Save and load models
- [Tutorial 4: Model Evaluation](04_evaluation.md) - Evaluate model quality
- [API Reference: Callbacks](../api/callbacks.md) - Complete callback API
