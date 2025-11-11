"""
Example 1: Basic GHSOM Training

This example demonstrates how to train a basic GHSOM model with synthetic data.
"""

import numpy as np
from ghsom import GHSOM

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic clustered data
print("Generating synthetic data...")
n_samples = 200
n_features = 10

# Create two clusters
cluster1 = np.random.randn(n_samples // 2, n_features) + [2, 2, 0, 0, 1, 1, 0, 0, 0.5, 0.5]
cluster2 = np.random.randn(n_samples // 2, n_features) + [-2, -2, 1, 1, 0, 0, 1, 1, -0.5, -0.5]
data = np.concatenate([cluster1, cluster2])

print(f"Dataset shape: {data.shape}")
print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")

# Initialize GHSOM
print("\nInitializing GHSOM...")
ghsom = GHSOM(
    input_dataset=data,
    t1=0.5,              # Growth threshold
    t2=0.05,             # Stopping criterion
    learning_rate=0.1,   # Initial learning rate
    decay=0.9,           # Decay rate
    gaussian_sigma=1.0   # Initial neighborhood size
)

# Train the model
print("Training GHSOM...")
model = ghsom.train(
    epochs_number=50,
    n_workers=-1  # Use all CPU cores
)

# Analyze results
print("\n" + "=" * 50)
print("Training Complete!")
print("=" * 50)

root_map = model.child_map
print(f"Root map shape: {root_map.map_shape()}")
print(f"Number of neurons in root map: {len(root_map.neurons)}")

# Check hierarchy depth
def get_hierarchy_depth(neuron, depth=0):
    """Calculate the maximum depth of the hierarchy."""
    if neuron.child_map is None:
        return depth
    max_depth = depth
    for child_neuron in neuron.child_map.neurons.values():
        child_depth = get_hierarchy_depth(child_neuron, depth + 1)
        max_depth = max(max_depth, child_depth)
    return max_depth

depth = get_hierarchy_depth(model)
print(f"Hierarchy depth: {depth}")

# Count total neurons
def count_neurons(neuron):
    """Count total neurons in the hierarchy."""
    count = 1
    if neuron.child_map is not None:
        for child_neuron in neuron.child_map.neurons.values():
            count += count_neurons(child_neuron)
    return count

total = count_neurons(model)
print(f"Total neurons: {total}")

# Show neurons with child maps
neurons_with_children = 0
for pos, neuron in root_map.neurons.items():
    if neuron.child_map is not None:
        neurons_with_children += 1
        print(f"  Neuron at {pos} has a child map of shape {neuron.child_map.map_shape()}")

print(f"Neurons with child maps: {neurons_with_children}/{len(root_map.neurons)}")

print("\nExample completed successfully!")
