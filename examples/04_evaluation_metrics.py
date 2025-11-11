"""
Example 4: Model Evaluation Metrics

This example demonstrates how to evaluate GHSOM models using various metrics.
"""

import numpy as np
from ghsom import GHSOM
from ghsom.evaluation.metrics import (
    mean_data_centroid_activation,
    hierarchy_depth,
    total_neurons_count,
)

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("GHSOM Model Evaluation Example")
print("=" * 60)

# Generate data with known clusters
print("\n1. Generating synthetic clustered data...")
n_samples_per_cluster = 100
n_features = 10

cluster1 = np.random.randn(n_samples_per_cluster, n_features) + [3, 3, 0, 0, 0, 0, 0, 0, 0, 0]
cluster2 = np.random.randn(n_samples_per_cluster, n_features) + [-3, -3, 0, 0, 0, 0, 0, 0, 0, 0]
cluster3 = np.random.randn(n_samples_per_cluster, n_features) + [0, 0, 3, 3, 0, 0, 0, 0, 0, 0]

data = np.concatenate([cluster1, cluster2, cluster3])
print(f"   Dataset shape: {data.shape}")
print(f"   Number of clusters: 3")

# Train model
print("\n2. Training GHSOM model...")
ghsom = GHSOM(
    input_dataset=data,
    t1=0.5,
    t2=0.05,
    learning_rate=0.1,
    decay=0.9,
    gaussian_sigma=1.0
)

model = ghsom.train(epochs_number=50, n_workers=-1)
print("   Training complete!")

# Evaluate using built-in metrics
print("\n3. Calculating evaluation metrics...")
print("-" * 60)

# Mean centroid activation
mean_act, std_act = mean_data_centroid_activation(model, data)
print(f"Mean Centroid Activation: {mean_act:.6f} ± {std_act:.6f}")

# Hierarchy depth
depth = hierarchy_depth(model)
print(f"Hierarchy Depth: {depth}")

# Total neurons
total_neurons = total_neurons_count(model)
print(f"Total Neurons: {total_neurons}")

# Custom evaluation: Analyze hierarchy structure
print("\n4. Analyzing hierarchy structure...")
print("-" * 60)


def analyze_hierarchy(neuron, level=0, stats=None):
    """Recursively analyze the hierarchy."""
    if stats is None:
        stats = {
            'neurons_per_level': {},
            'neurons_with_children': 0,
            'leaf_neurons': 0
        }

    # Update level count
    if level not in stats['neurons_per_level']:
        stats['neurons_per_level'][level] = 0
    stats['neurons_per_level'][level] += 1

    # Check if neuron has children
    if neuron.child_map is not None:
        stats['neurons_with_children'] += 1
        for child in neuron.child_map.neurons.values():
            analyze_hierarchy(child, level + 1, stats)
    else:
        stats['leaf_neurons'] += 1

    return stats


stats = analyze_hierarchy(model)

print("Neurons per Level:")
for level in sorted(stats['neurons_per_level'].keys()):
    count = stats['neurons_per_level'][level]
    print(f"  Level {level}: {count:3d} neurons")

print(f"\nNeurons with children: {stats['neurons_with_children']}")
print(f"Leaf neurons: {stats['leaf_neurons']}")

# Custom evaluation: Quantization error per sample
print("\n5. Calculating quantization error distribution...")
print("-" * 60)


def calculate_qe_distribution(model, data):
    """Calculate quantization error for each sample."""
    errors = []

    for sample in data:
        # Find BMU
        neuron = model
        while neuron.child_map is not None:
            gsom_map = neuron.child_map
            winner = gsom_map.winner_neuron(sample)[0][0]
            neuron = winner

        # Calculate error
        error = neuron.activation(sample)
        errors.append(error)

    return np.array(errors)


qe_errors = calculate_qe_distribution(model, data)

print(f"Quantization Error Statistics:")
print(f"  Mean:   {np.mean(qe_errors):.6f}")
print(f"  Median: {np.median(qe_errors):.6f}")
print(f"  Std:    {np.std(qe_errors):.6f}")
print(f"  Min:    {np.min(qe_errors):.6f}")
print(f"  Max:    {np.max(qe_errors):.6f}")

# Percentiles
percentiles = [25, 50, 75, 90, 95, 99]
print(f"\nPercentiles:")
for p in percentiles:
    val = np.percentile(qe_errors, p)
    print(f"  {p}th: {val:.6f}")

# Custom evaluation: Neuron usage
print("\n6. Analyzing neuron usage...")
print("-" * 60)


def analyze_neuron_usage(model, data):
    """Analyze how data is distributed across neurons."""
    from collections import defaultdict
    usage = defaultdict(int)

    for sample in data:
        neuron = model
        path = []

        while neuron.child_map is not None:
            gsom_map = neuron.child_map
            winner = gsom_map.winner_neuron(sample)[0][0]
            path.append(str(winner.position))
            neuron = winner

        usage[tuple(path)] += 1

    return usage


usage = analyze_neuron_usage(model, data)

print(f"Unique leaf neurons used: {len(usage)}")
print(f"Total samples: {len(data)}")
print(f"Average samples per neuron: {len(data) / len(usage):.2f}")

# Show top 5 most used neurons
top_neurons = sorted(usage.items(), key=lambda x: x[1], reverse=True)[:5]
print(f"\nTop 5 most used neurons:")
for i, (path, count) in enumerate(top_neurons, 1):
    percentage = (count / len(data)) * 100
    path_str = " -> ".join(path)
    print(f"  {i}. Path [{path_str}]: {count:3d} samples ({percentage:5.1f}%)")

# Summary report
print("\n" + "=" * 60)
print("Evaluation Summary")
print("=" * 60)
print(f"Model Configuration:")
print(f"  t1: 0.5, t2: 0.05")
print(f"\nDataset:")
print(f"  Samples: {len(data)}")
print(f"  Features: {data.shape[1]}")
print(f"  Clusters: 3")
print(f"\nModel Structure:")
print(f"  Hierarchy Depth: {depth}")
print(f"  Total Neurons: {total_neurons}")
print(f"  Leaf Neurons: {stats['leaf_neurons']}")
print(f"\nPerformance:")
print(f"  Mean Activation: {mean_act:.6f}")
print(f"  Mean QE: {np.mean(qe_errors):.6f}")
print(f"  Neuron Usage: {len(usage)} / {stats['leaf_neurons']} leaf neurons used")
print("=" * 60)

print("\nExample completed successfully!")
