"""
Example 5: Multiprocessing and Performance Optimization

This example demonstrates parallel training and performance optimization techniques.
"""

import numpy as np
import time
import os
from ghsom import GHSOM

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("GHSOM Multiprocessing Example")
print("=" * 60)

# System info
print(f"\nSystem Information:")
print(f"  Available CPU cores: {os.cpu_count()}")

# Generate a moderately large dataset
print("\nGenerating dataset...")
n_samples = 500
n_features = 20
data = np.random.rand(n_samples, n_features)
print(f"  Dataset shape: {data.shape}")

# Configuration
config = {
    "t1": 0.5,
    "t2": 0.05,
    "learning_rate": 0.1,
    "decay": 0.9,
    "gaussian_sigma": 1.0,
    "epochs": 30
}

# Compare different worker configurations
worker_configs = [
    ("Single worker", 1),
    ("Two workers", 2),
    ("Four workers", 4),
    ("All cores", -1)
]

results = []

print("\n" + "=" * 60)
print("Training with Different Worker Configurations")
print("=" * 60)

for name, n_workers in worker_configs:
    print(f"\n{name} (n_workers={n_workers}):")
    print("-" * 40)

    # Train model
    start_time = time.time()

    ghsom = GHSOM(
        input_dataset=data,
        t1=config["t1"],
        t2=config["t2"],
        learning_rate=config["learning_rate"],
        decay=config["decay"],
        gaussian_sigma=config["gaussian_sigma"]
    )

    model = ghsom.train(
        epochs_number=config["epochs"],
        n_workers=n_workers,
        seed=42  # Same seed for fair comparison
    )

    elapsed = time.time() - start_time

    # Calculate stats
    def count_neurons(neuron):
        count = 1
        if neuron.child_map is not None:
            for child in neuron.child_map.neurons.values():
                count += count_neurons(child)
        return count

    total_neurons = count_neurons(model)
    samples_per_second = (n_samples * config["epochs"]) / elapsed

    print(f"  Training time: {elapsed:.2f} seconds")
    print(f"  Throughput: {samples_per_second:.0f} samples/second")
    print(f"  Total neurons: {total_neurons}")

    results.append({
        'name': name,
        'n_workers': n_workers,
        'elapsed': elapsed,
        'throughput': samples_per_second,
        'neurons': total_neurons
    })

# Performance comparison
print("\n" + "=" * 60)
print("Performance Comparison")
print("=" * 60)

baseline = results[0]['elapsed']  # Single worker time

print(f"\n{'Configuration':<20} {'Time (s)':<12} {'Speedup':<10} {'Throughput':<20}")
print("-" * 70)

for r in results:
    speedup = baseline / r['elapsed']
    print(f"{r['name']:<20} {r['elapsed']:>8.2f}     {speedup:>6.2f}x    "
          f"{r['throughput']:>10.0f} samples/s")

# Find best configuration
best = min(results, key=lambda x: x['elapsed'])
print(f"\nBest configuration: {best['name']} ({best['elapsed']:.2f}s)")
print(f"Speedup over single worker: {baseline / best['elapsed']:.2f}x")

# Memory optimization example
print("\n" + "=" * 60)
print("Memory Optimization with Dataset Sampling")
print("=" * 60)

print("\nTraining with different dataset_percentage values...")

for percentage in [0.25, 0.5, 1.0]:
    print(f"\nDataset percentage: {percentage*100:.0f}%")
    print("-" * 40)

    start_time = time.time()

    ghsom = GHSOM(
        input_dataset=data,
        t1=config["t1"],
        t2=config["t2"],
        learning_rate=config["learning_rate"],
        decay=config["decay"],
        gaussian_sigma=config["gaussian_sigma"]
    )

    model = ghsom.train(
        epochs_number=config["epochs"],
        dataset_percentage=percentage,
        n_workers=-1,
        seed=42
    )

    elapsed = time.time() - start_time
    total_neurons = count_neurons(model)

    print(f"  Training time: {elapsed:.2f} seconds")
    print(f"  Total neurons: {total_neurons}")
    print(f"  Effective samples per epoch: {int(n_samples * percentage)}")

# Best practices guide
print("\n" + "=" * 60)
print("Best Practices for Performance Optimization")
print("=" * 60)

print("""
1. Worker Configuration:
   - Small datasets (< 1000 samples): Use 1-2 workers
   - Medium datasets (1000-10000): Use half of available cores
   - Large datasets (> 10000): Use all available cores (-1)

2. Dataset Sampling:
   - Use dataset_percentage=0.25 for memory efficiency
   - Use dataset_percentage=1.0 for maximum accuracy
   - Balance based on your memory constraints

3. Reproducibility:
   - Always set seed parameter for reproducible results
   - Note: n_workers > 1 may introduce some non-determinism

4. Monitoring:
   - Use callbacks to monitor training progress
   - Track time and memory usage for your specific use case

5. Hardware Considerations:
   - More workers ≠ always faster (overhead exists)
   - Test different configurations with your data
   - Consider I/O bottlenecks for very large datasets
""")

# Recommended configuration
print("=" * 60)
print("Recommended Configuration for This Dataset")
print("=" * 60)

if len(data) < 1000:
    rec_workers = 1
    reason = "Small dataset, single worker avoids overhead"
elif len(data) < 10000:
    rec_workers = max(1, os.cpu_count() // 2)
    reason = "Medium dataset, use half of available cores"
else:
    rec_workers = -1
    reason = "Large dataset, use all available cores"

print(f"\nDataset size: {len(data)} samples")
print(f"Recommended n_workers: {rec_workers}")
print(f"Reason: {reason}")
print(f"Recommended dataset_percentage: 0.25-0.5 (balance speed/accuracy)")

print("\nExample completed successfully!")
