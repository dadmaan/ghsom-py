# Tutorial 5: Advanced Usage

Advanced techniques for optimizing and deploying GHSOM models.

## What You'll Learn

- Parallel processing optimization
- Custom metrics and growing strategies
- Memory optimization techniques
- Production deployment patterns
- Hyperparameter tuning strategies

## Parallel Processing Optimization

### Understanding n_workers

```python
import numpy as np
import time
from ghsom import GHSOM

data = np.random.rand(500, 20)

# Test different worker configurations
worker_configs = [1, 2, 4, -1]  # -1 means all CPU cores

for n_workers in worker_configs:
    start = time.time()

    ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
    model = ghsom.train(epochs_number=30, n_workers=n_workers)

    elapsed = time.time() - start
    workers_used = n_workers if n_workers > 0 else "all"
    print(f"Workers: {workers_used:>3} | Time: {elapsed:6.2f}s")
```

### Best Practices for Parallelization

```python
import os
from ghsom import GHSOM

# For small datasets (< 1000 samples)
# Single worker might be faster due to overhead
if len(data) < 1000:
    n_workers = 1
# For medium datasets (1000-10000 samples)
# Use half of available cores
elif len(data) < 10000:
    n_workers = max(1, os.cpu_count() // 2)
# For large datasets (> 10000 samples)
# Use all available cores
else:
    n_workers = -1

ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
model = ghsom.train(epochs_number=50, n_workers=n_workers)
```

## Custom Growing Metrics

GHSOM supports two growing metrics: `qe` and `mqe`:

```python
import numpy as np
from ghsom import GHSOM

data = np.random.rand(200, 10)

# Using Quantization Error (default)
ghsom_qe = GHSOM(
    input_dataset=data,
    t1=0.5,
    t2=0.05,
    growing_metric="qe"  # Default
)
model_qe = ghsom_qe.train(epochs_number=50)

# Using Mean Quantization Error
ghsom_mqe = GHSOM(
    input_dataset=data,
    t1=0.5,
    t2=0.05,
    growing_metric="mqe"
)
model_mqe = ghsom_mqe.train(epochs_number=50)

print(f"QE model depth: {hierarchy_depth(model_qe)}")
print(f"MQE model depth: {hierarchy_depth(model_mqe)}")
```

**When to use each:**
- **QE (Quantization Error)**: Better for capturing fine-grained details
- **MQE (Mean Quantization Error)**: More balanced, less sensitive to outliers

## Memory Optimization

### Training with Dataset Sampling

```python
import numpy as np
from ghsom import GHSOM

# Large dataset
large_data = np.random.rand(10000, 50)

# Use dataset_percentage to reduce memory
ghsom = GHSOM(
    input_dataset=large_data,
    t1=0.5,
    t2=0.05,
    learning_rate=0.1,
    decay=0.9,
    gaussian_sigma=1.0
)

# Only use 25% of data per training iteration
# This reduces memory usage significantly
model = ghsom.train(
    epochs_number=50,
    dataset_percentage=0.25,  # Use 25% of data
    min_dataset_size=10       # Minimum samples required
)
```

### Removing Training Data Post-Training

```python
def clean_model(neuron):
    """Remove training data from model to save memory."""
    # Clear training data
    neuron.input_dataset = None

    # Recursively clean children
    if neuron.child_map is not None:
        neuron.child_map.input_dataset = None
        for child in neuron.child_map.neurons.values():
            clean_model(child)

# After training
model = ghsom.train(epochs_number=50)

# Clean before saving
clean_model(model)
save_model(model, 'lightweight_model.pkl')
```

**Warning**: Don't clean if you need to evaluate with `mean_data_centroid_activation`!

## Hyperparameter Tuning

### Grid Search

```python
import numpy as np
from itertools import product
from ghsom import GHSOM
from ghsom.evaluation.metrics import mean_data_centroid_activation

def grid_search(data, param_grid):
    """Perform grid search over parameters."""
    results = []

    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()

    for combination in product(*values):
        params = dict(zip(keys, combination))

        print(f"Testing {params}...")

        # Train model
        ghsom = GHSOM(
            input_dataset=data,
            t1=params['t1'],
            t2=params['t2'],
            learning_rate=params['learning_rate'],
            decay=params['decay'],
            gaussian_sigma=params['gaussian_sigma']
        )
        model = ghsom.train(epochs_number=30, n_workers=-1)

        # Evaluate
        mean_act, _ = mean_data_centroid_activation(model, data)

        results.append({
            'params': params,
            'score': mean_act
        })

    return results

# Define parameter grid
param_grid = {
    't1': [0.3, 0.5, 0.7],
    't2': [0.03, 0.05, 0.07],
    'learning_rate': [0.1],
    'decay': [0.9],
    'gaussian_sigma': [1.0, 1.5]
}

# Run grid search
data = np.random.rand(200, 10)
results = grid_search(data, param_grid)

# Find best parameters
best = min(results, key=lambda x: x['score'])
print(f"\nBest parameters: {best['params']}")
print(f"Best score: {best['score']:.4f}")
```

### Random Search

```python
import numpy as np
from ghsom import GHSOM
from ghsom.evaluation.metrics import mean_data_centroid_activation

def random_search(data, n_iterations=20):
    """Perform random search over parameters."""
    results = []

    for i in range(n_iterations):
        # Sample random parameters
        params = {
            't1': np.random.uniform(0.3, 0.7),
            't2': np.random.uniform(0.02, 0.08),
            'learning_rate': np.random.uniform(0.05, 0.2),
            'decay': np.random.uniform(0.85, 0.95),
            'gaussian_sigma': np.random.uniform(0.8, 2.0)
        }

        print(f"Iteration {i+1}/{n_iterations}: {params}")

        # Train and evaluate
        ghsom = GHSOM(
            input_dataset=data,
            t1=params['t1'],
            t2=params['t2'],
            learning_rate=params['learning_rate'],
            decay=params['decay'],
            gaussian_sigma=params['gaussian_sigma']
        )
        model = ghsom.train(epochs_number=30, n_workers=-1)

        mean_act, _ = mean_data_centroid_activation(model, data)

        results.append({
            'params': params,
            'score': mean_act
        })

    return results

# Run random search
data = np.random.rand(200, 10)
results = random_search(data, n_iterations=10)

# Find best
best = min(results, key=lambda x: x['score'])
print(f"\nBest parameters: {best['params']}")
print(f"Best score: {best['score']:.4f}")
```

## Production Deployment Patterns

### Online Prediction Service

```python
from ghsom.io import load_model
import numpy as np

class GHSOMPredictor:
    """Production-ready GHSOM predictor."""

    def __init__(self, model_path):
        """Load the trained model."""
        self.model = load_model(model_path)

    def predict(self, samples):
        """Find BMUs for samples."""
        if len(samples.shape) == 1:
            samples = samples.reshape(1, -1)

        predictions = []

        for sample in samples:
            neuron = self.model
            path = []

            # Traverse hierarchy
            while neuron.child_map is not None:
                gsom_map = neuron.child_map
                winner = gsom_map.winner_neuron(sample)[0][0]
                path.append(winner.position)
                neuron = winner

            predictions.append({
                'path': path,
                'position': neuron.position,
                'distance': neuron.activation(sample)
            })

        return predictions

    def predict_batch(self, samples, batch_size=100):
        """Process samples in batches."""
        results = []

        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            batch_results = self.predict(batch)
            results.extend(batch_results)

        return results

# Usage
predictor = GHSOMPredictor('production_model.pkl')

# Single prediction
sample = np.random.rand(10)
result = predictor.predict(sample)
print(f"Prediction: {result}")

# Batch prediction
samples = np.random.rand(1000, 10)
results = predictor.predict_batch(samples, batch_size=100)
```

### Model Versioning

```python
from pathlib import Path
import json
from datetime import datetime
from ghsom.io import save_model

class ModelRegistry:
    """Manage model versions."""

    def __init__(self, registry_path='model_registry'):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)

    def save_version(self, model, version_name, metadata=None):
        """Save a versioned model."""
        version_dir = self.registry_path / version_name
        version_dir.mkdir(exist_ok=True)

        # Save model
        model_path = version_dir / 'model.pkl'
        save_model(model, str(model_path))

        # Save metadata
        metadata = metadata or {}
        metadata['version'] = version_name
        metadata['saved_at'] = datetime.now().isoformat()

        with open(version_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved version: {version_name}")

    def load_version(self, version_name):
        """Load a specific version."""
        version_dir = self.registry_path / version_name
        model_path = version_dir / 'model.pkl'

        model = load_model(str(model_path))

        with open(version_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        return model, metadata

    def list_versions(self):
        """List all available versions."""
        versions = []
        for version_dir in self.registry_path.iterdir():
            if version_dir.is_dir():
                with open(version_dir / 'metadata.json', 'r') as f:
                    metadata = json.load(f)
                versions.append(metadata)
        return versions

# Usage
registry = ModelRegistry()

# Save model versions
registry.save_version(
    model,
    'v1.0.0',
    metadata={'t1': 0.5, 't2': 0.05, 'accuracy': 0.85}
)

# List versions
versions = registry.list_versions()
for v in versions:
    print(f"Version {v['version']}: saved at {v['saved_at']}")

# Load specific version
model, metadata = registry.load_version('v1.0.0')
```

## Reproducibility

### Setting Random Seeds

```python
import numpy as np
from ghsom import GHSOM

def train_reproducible(data, config, seed=42):
    """Train with reproducible results."""
    # Set numpy seed
    np.random.seed(seed)

    ghsom = GHSOM(
        input_dataset=data,
        t1=config['t1'],
        t2=config['t2'],
        learning_rate=config['learning_rate'],
        decay=config['decay'],
        gaussian_sigma=config['gaussian_sigma']
    )

    # Pass seed to training
    model = ghsom.train(
        epochs_number=config['epochs'],
        seed=seed,  # This ensures reproducible weight initialization
        n_workers=1  # Parallel processing can introduce non-determinism
    )

    return model

# Train twice with same seed - should get identical results
config = {'t1': 0.5, 't2': 0.05, 'learning_rate': 0.1,
          'decay': 0.9, 'gaussian_sigma': 1.0, 'epochs': 30}

data = np.random.rand(100, 10)

model1 = train_reproducible(data, config, seed=42)
model2 = train_reproducible(data, config, seed=42)

# Verify they're the same
print("Models should be identical with same seed")
```

## Complete Advanced Example

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ghsom import GHSOM
from ghsom.io import save_model
from ghsom.evaluation.metrics import mean_data_centroid_activation

# 1. Data preparation
data = np.random.rand(1000, 20)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Normalize
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# 2. Optimized training
ghsom = GHSOM(
    input_dataset=train_data,
    t1=0.5,
    t2=0.05,
    learning_rate=0.1,
    decay=0.9,
    gaussian_sigma=1.0
)

model = ghsom.train(
    epochs_number=50,
    dataset_percentage=0.25,  # Memory optimization
    n_workers=-1,             # Use all cores
    seed=42                   # Reproducibility
)

# 3. Evaluation
train_score, _ = mean_data_centroid_activation(model, train_data)
test_score, _ = mean_data_centroid_activation(model, test_data)

print(f"Train score: {train_score:.4f}")
print(f"Test score: {test_score:.4f}")

# 4. Production deployment
clean_model(model)  # Remove training data
save_model(model, 'production/model_v1.pkl')

print("Model ready for production!")
```

## Next Steps

- [API Reference](../api/core/ghsom.md) - Complete API documentation
- [Examples Repository](https://github.com/dadmaan/ghsom-py/tree/main/examples) - More examples
- [Contributing Guide](../contributing.md) - Contribute to the project
