"""
Example 2: Training with Callbacks

This example demonstrates how to use callbacks to monitor training progress.
"""

import numpy as np
import time
from ghsom import GHSOM
from ghsom.callbacks import TrackingCallback

# Set random seed for reproducibility
np.random.seed(42)


class DetailedCallback(TrackingCallback):
    """Custom callback for detailed training monitoring."""

    def __init__(self):
        self.start_time = None
        self.maps_created = 0
        self.qe_values = []

    def on_train_begin(self, config):
        """Called when training starts."""
        self.start_time = time.time()
        self.maps_created = 0
        self.qe_values = []

        print("=" * 60)
        print("GHSOM Training Started")
        print("=" * 60)
        print("\nConfiguration:")
        for key, value in config.items():
            print(f"  {key:20s}: {value}")
        print()

    def on_map_created(self, metrics):
        """Called when a new map is created."""
        self.maps_created += 1
        qe = metrics.get('qe', metrics.get('mqe'))
        position = metrics.get('neuron_position', 'unknown')

        if qe is not None:
            self.qe_values.append(qe)

        # Print progress every 5 maps
        if self.maps_created % 5 == 0:
            elapsed = time.time() - self.start_time
            avg_qe = np.mean(self.qe_values[-5:]) if len(self.qe_values) >= 5 else np.mean(self.qe_values)
            print(f"[{elapsed:6.2f}s] Maps: {self.maps_created:3d} | "
                  f"Avg QE (last 5): {avg_qe:.6f} | "
                  f"Position: {position}")

    def on_train_end(self, results):
        """Called when training completes."""
        elapsed = time.time() - self.start_time

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Total maps created: {self.maps_created}")
        print(f"Total neurons: {results['total_neurons']}")

        if self.qe_values:
            print(f"\nQuantization Error Statistics:")
            print(f"  Mean: {np.mean(self.qe_values):.6f}")
            print(f"  Std:  {np.std(self.qe_values):.6f}")
            print(f"  Min:  {np.min(self.qe_values):.6f}")
            print(f"  Max:  {np.max(self.qe_values):.6f}")


# Generate data
print("Generating synthetic data...")
data = np.random.rand(300, 12)

# Initialize GHSOM
ghsom = GHSOM(
    input_dataset=data,
    t1=0.5,
    t2=0.05,
    learning_rate=0.1,
    decay=0.9,
    gaussian_sigma=1.0
)

# Train with callback
model = ghsom.train(
    epochs_number=50,
    n_workers=-1,
    callbacks=[DetailedCallback()]
)

print("\nExample completed successfully!")
