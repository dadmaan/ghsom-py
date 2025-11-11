# Tutorial 4: Model Evaluation

Learn how to evaluate and analyze your trained GHSOM models using built-in metrics.

## What You'll Learn

- How to calculate evaluation metrics
- How to interpret model quality
- How to compare different models
- How to analyze the hierarchy structure

## Available Metrics

GHSOM-Py provides several evaluation metrics:

- **Quantization Error (QE)**: Average distance between data points and their BMUs
- **Mean Data Centroid Activation**: Average activation of centroid neurons
- **Hierarchy Depth**: Maximum depth of the hierarchy
- **Total Neurons**: Total number of neurons in the hierarchy
- **Neuron Usage**: How many neurons are actually used

## Basic Evaluation

```python
import numpy as np
from ghsom import GHSOM
from ghsom.evaluation.metrics import (
    mean_data_centroid_activation,
    hierarchy_depth,
    total_neurons_count
)

# Train a model
data = np.random.rand(200, 10)
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
model = ghsom.train(epochs_number=50)

# Calculate metrics
mean_act, std_act = mean_data_centroid_activation(model, data)
depth = hierarchy_depth(model)
total_neurons = total_neurons_count(model)

print("Model Evaluation Metrics:")
print(f"  Mean Centroid Activation: {mean_act:.4f} ± {std_act:.4f}")
print(f"  Hierarchy Depth: {depth}")
print(f"  Total Neurons: {total_neurons}")
```

## Understanding Quantization Error

Quantization error measures how well the model represents the data:

```python
from ghsom import GHSOM
import numpy as np

def calculate_quantization_error(model, data):
    """Calculate average quantization error for a dataset."""
    errors = []

    for sample in data:
        # Find the best matching unit
        neuron = model
        while neuron.child_map is not None:
            gsom_map = neuron.child_map
            winner = gsom_map.winner_neuron(sample)[0][0]
            neuron = winner

        # Calculate distance to BMU
        error = neuron.activation(sample)
        errors.append(error)

    return np.mean(errors), np.std(errors)

# Evaluate
data = np.random.rand(200, 10)
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
model = ghsom.train(epochs_number=50)

qe_mean, qe_std = calculate_quantization_error(model, data)
print(f"Quantization Error: {qe_mean:.4f} ± {qe_std:.4f}")
```

Lower quantization error indicates better data representation.

## Analyzing Hierarchy Structure

```python
from ghsom import GHSOM
import numpy as np

def analyze_hierarchy(neuron, level=0, stats=None):
    """Recursively analyze the hierarchy structure."""
    if stats is None:
        stats = {
            'neurons_per_level': {},
            'max_depth': 0,
            'total_neurons': 0,
            'neurons_with_children': 0
        }

    # Update stats
    stats['total_neurons'] += 1
    stats['max_depth'] = max(stats['max_depth'], level)

    if level not in stats['neurons_per_level']:
        stats['neurons_per_level'][level] = 0
    stats['neurons_per_level'][level] += 1

    # Check children
    if neuron.child_map is not None:
        stats['neurons_with_children'] += 1
        for child in neuron.child_map.neurons.values():
            analyze_hierarchy(child, level + 1, stats)

    return stats

# Analyze
data = np.random.rand(200, 10)
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
model = ghsom.train(epochs_number=50)

stats = analyze_hierarchy(model)

print("Hierarchy Analysis:")
print(f"  Max Depth: {stats['max_depth']}")
print(f"  Total Neurons: {stats['total_neurons']}")
print(f"  Neurons with Children: {stats['neurons_with_children']}")
print("\nNeurons per Level:")
for level, count in sorted(stats['neurons_per_level'].items()):
    print(f"  Level {level}: {count} neurons")
```

## Comparing Models

Compare different configurations to find the best model:

```python
import numpy as np
from ghsom import GHSOM
from ghsom.evaluation.metrics import mean_data_centroid_activation, hierarchy_depth

def evaluate_config(data, t1, t2, epochs=50):
    """Train and evaluate a GHSOM configuration."""
    ghsom = GHSOM(
        input_dataset=data,
        t1=t1,
        t2=t2,
        learning_rate=0.1,
        decay=0.9,
        gaussian_sigma=1.0
    )
    model = ghsom.train(epochs_number=epochs, n_workers=-1)

    # Calculate metrics
    mean_act, std_act = mean_data_centroid_activation(model, data)
    depth = hierarchy_depth(model)

    return {
        't1': t1,
        't2': t2,
        'mean_activation': mean_act,
        'std_activation': std_act,
        'depth': depth
    }

# Compare configurations
data = np.random.rand(200, 10)
configs = [
    (0.3, 0.05),
    (0.5, 0.05),
    (0.7, 0.05),
    (0.5, 0.03),
    (0.5, 0.07)
]

results = []
for t1, t2 in configs:
    print(f"Evaluating t1={t1}, t2={t2}...")
    result = evaluate_config(data, t1, t2)
    results.append(result)

# Print comparison
print("\nModel Comparison:")
print(f"{'t1':<6} {'t2':<6} {'Mean Act':<12} {'Std Act':<12} {'Depth':<6}")
print("-" * 50)
for r in results:
    print(f"{r['t1']:<6.2f} {r['t2']:<6.2f} {r['mean_activation']:<12.4f} "
          f"{r['std_activation']:<12.4f} {r['depth']:<6}")
```

## Neuron Usage Analysis

Analyze how data is distributed across neurons:

```python
from collections import defaultdict
import numpy as np

def analyze_neuron_usage(model, data):
    """Analyze how data points are distributed across neurons."""
    usage = defaultdict(int)

    for sample in data:
        # Find BMU for this sample
        neuron = model
        path = []

        while neuron.child_map is not None:
            gsom_map = neuron.child_map
            winner = gsom_map.winner_neuron(sample)[0][0]
            path.append(str(winner.position))
            neuron = winner

        # Record the leaf neuron
        usage[tuple(path)] += 1

    return usage

# Analyze usage
data = np.random.rand(200, 10)
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
model = ghsom.train(epochs_number=50)

usage = analyze_neuron_usage(model, data)

print(f"Number of unique leaf neurons used: {len(usage)}")
print(f"Total data points: {len(data)}")
print(f"Average points per neuron: {len(data) / len(usage):.2f}")

# Show top 5 most used neurons
top_neurons = sorted(usage.items(), key=lambda x: x[1], reverse=True)[:5]
print("\nTop 5 most used neurons:")
for path, count in top_neurons:
    print(f"  Path {' -> '.join(path)}: {count} samples ({count/len(data)*100:.1f}%)")
```

## Cross-Validation

Evaluate model generalization:

```python
import numpy as np
from sklearn.model_selection import KFold
from ghsom import GHSOM

def cross_validate_ghsom(data, t1, t2, n_splits=5):
    """Perform k-fold cross-validation."""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
        print(f"Fold {fold + 1}/{n_splits}...")

        train_data = data[train_idx]
        val_data = data[val_idx]

        # Train on fold
        ghsom = GHSOM(
            input_dataset=train_data,
            t1=t1,
            t2=t2,
            learning_rate=0.1,
            decay=0.9,
            gaussian_sigma=1.0
        )
        model = ghsom.train(epochs_number=30)

        # Evaluate on validation set
        mean_act, std_act = mean_data_centroid_activation(model, val_data)
        scores.append(mean_act)

    return np.mean(scores), np.std(scores)

# Run cross-validation
data = np.random.rand(200, 10)
mean_score, std_score = cross_validate_ghsom(data, t1=0.5, t2=0.05)

print(f"\nCross-Validation Results:")
print(f"  Mean Score: {mean_score:.4f} ± {std_score:.4f}")
```

## Performance Benchmarking

Measure training time and memory usage:

```python
import time
import numpy as np
from ghsom import GHSOM

def benchmark_training(data, t1, t2, epochs=50):
    """Benchmark training performance."""
    start_time = time.time()

    ghsom = GHSOM(
        input_dataset=data,
        t1=t1,
        t2=t2,
        learning_rate=0.1,
        decay=0.9,
        gaussian_sigma=1.0
    )
    model = ghsom.train(epochs_number=epochs, n_workers=-1)

    elapsed = time.time() - start_time

    # Calculate stats
    stats = analyze_hierarchy(model)

    return {
        'elapsed_time': elapsed,
        'samples_per_second': len(data) * epochs / elapsed,
        'total_neurons': stats['total_neurons'],
        'max_depth': stats['max_depth']
    }

# Benchmark
data = np.random.rand(500, 20)
results = benchmark_training(data, t1=0.5, t2=0.05, epochs=50)

print("Benchmark Results:")
print(f"  Training Time: {results['elapsed_time']:.2f} seconds")
print(f"  Samples/Second: {results['samples_per_second']:.0f}")
print(f"  Total Neurons: {results['total_neurons']}")
print(f"  Max Depth: {results['max_depth']}")
```

## Complete Evaluation Report

```python
import numpy as np
from ghsom import GHSOM
from ghsom.evaluation.metrics import mean_data_centroid_activation, hierarchy_depth, total_neurons_count

def generate_evaluation_report(model, data, config):
    """Generate a comprehensive evaluation report."""
    print("=" * 60)
    print("GHSOM Model Evaluation Report")
    print("=" * 60)

    # Configuration
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Dataset info
    print(f"\nDataset:")
    print(f"  Samples: {len(data)}")
    print(f"  Features: {data.shape[1]}")

    # Metrics
    mean_act, std_act = mean_data_centroid_activation(model, data)
    depth = hierarchy_depth(model)
    total_neurons = total_neurons_count(model)

    print(f"\nMetrics:")
    print(f"  Mean Centroid Activation: {mean_act:.4f} ± {std_act:.4f}")
    print(f"  Hierarchy Depth: {depth}")
    print(f"  Total Neurons: {total_neurons}")

    # Hierarchy structure
    stats = analyze_hierarchy(model)
    print(f"\nHierarchy Structure:")
    for level, count in sorted(stats['neurons_per_level'].items()):
        print(f"  Level {level}: {count} neurons")

    print("=" * 60)

# Usage
data = np.random.rand(200, 10)
config = {
    "t1": 0.5,
    "t2": 0.05,
    "learning_rate": 0.1,
    "decay": 0.9,
    "gaussian_sigma": 1.0,
    "epochs": 50
}

ghsom = GHSOM(
    input_dataset=data,
    t1=config["t1"],
    t2=config["t2"],
    learning_rate=config["learning_rate"],
    decay=config["decay"],
    gaussian_sigma=config["gaussian_sigma"]
)
model = ghsom.train(epochs_number=config["epochs"])

generate_evaluation_report(model, data, config)
```

## Next Steps

- [Tutorial 5: Advanced Usage](05_advanced.md) - Advanced techniques
- [API Reference: Evaluation](../api/evaluation.md) - Complete evaluation API
- [Examples](https://github.com/dadmaan/ghsom-py/tree/main/examples) - More evaluation examples
