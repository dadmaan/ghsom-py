# Quick Start

This guide will help you train your first GHSOM model in just a few minutes.

## Basic Usage

```python
import numpy as np
from ghsom import GHSOM

# Create sample data (100 samples, 10 features)
data = np.random.rand(100, 10)

# Initialize GHSOM
ghsom = GHSOM(
    input_dataset=data,
    t1=0.5,              # Growth threshold
    t2=0.05,             # Stopping criterion
    learning_rate=0.1,
    decay=0.9,
    gaussian_sigma=1.0
)

# Train the model
model = ghsom.train(
    epochs_number=50,
    n_workers=-1  # Use all CPU cores
)

print(f"Training complete!")
print(f"Root map shape: {model.child_map.map_shape()}")
```

## Understanding the Parameters

### Core Parameters

- **input_dataset**: Your training data as a NumPy array (shape: `[n_samples, n_features]`)
- **t1**: Growth threshold (0.0-1.0, typical: 0.3-0.7)
    - Smaller values create larger maps with more detail
    - Larger values create smaller, more compact maps
- **t2**: Stopping criterion (0.0-1.0, typical: 0.01-0.1)
    - Smaller values create deeper hierarchies
    - Larger values create shallower hierarchies

### Training Parameters

- **learning_rate**: Initial learning rate (typical: 0.05-0.3)
- **decay**: How quickly learning rate decreases (typical: 0.8-0.95)
- **gaussian_sigma**: Initial neighborhood size (typical: 1.0-3.0)

### Advanced Parameters

- **epochs_number**: Number of training epochs (default: 15)
- **n_workers**: Number of CPU cores to use (-1 for all cores)
- **growing_metric**: Quality metric ("qe" or "mqe", default: "qe")

## Working with Real Data

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from ghsom import GHSOM

# Load a real dataset
iris = load_iris()
data = iris.data

# Standardize features (recommended)
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Train GHSOM
ghsom = GHSOM(
    input_dataset=data,
    t1=0.4,
    t2=0.05,
    learning_rate=0.1,
    decay=0.9,
    gaussian_sigma=1.5
)

model = ghsom.train(epochs_number=30)

print(f"Trained on {len(data)} samples with {data.shape[1]} features")
```

## Exploring the Trained Model

```python
# Access the root neuron
root = model

# Access the root map (2x2 initial map)
root_map = root.child_map

# Get map dimensions
print(f"Root map shape: {root_map.map_shape()}")

# Access individual neurons
for position, neuron in root_map.neurons.items():
    print(f"Neuron at {position}:")
    print(f"  - Weight vector shape: {neuron.weights.shape}")
    print(f"  - Has child map: {neuron.child_map is not None}")
    print(f"  - Dataset size: {len(neuron.input_dataset)}")
```

## Saving Your Model

```python
from ghsom.io import save_model, load_model

# Save the trained model
save_model(model, 'my_ghsom_model.pkl')

# Load it later
loaded_model = load_model('my_ghsom_model.pkl')
```

## Using Callbacks for Tracking

```python
from ghsom import GHSOM
from ghsom.callbacks import TrackingCallback

class PrintCallback(TrackingCallback):
    def on_train_begin(self, config):
        print(f"Training started with config: {config}")

    def on_map_created(self, metrics):
        print(f"New map created: QE={metrics.get('qe', 'N/A')}")

    def on_train_end(self, results):
        print(f"Training complete: {results}")

# Train with callback
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
model = ghsom.train(
    epochs_number=50,
    callbacks=[PrintCallback()]
)
```

## Next Steps

Now that you've trained your first model, explore:

- [Using Callbacks](../tutorials/02_callbacks.md) - Track training progress
- [Model Evaluation](../tutorials/04_evaluation.md) - Evaluate model quality
- [Advanced Usage](../tutorials/05_advanced.md) - Custom metrics and optimization
- [API Reference](../api/core/ghsom.md) - Complete API documentation
