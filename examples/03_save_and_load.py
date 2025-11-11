"""
Example 3: Saving and Loading Models

This example demonstrates how to save trained models and load them later.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from ghsom import GHSOM
from ghsom.io import save_model, load_model

# Set random seed for reproducibility
np.random.seed(42)

# Create experiment directory
exp_dir = Path("experiment_output")
exp_dir.mkdir(exist_ok=True)

print("=" * 60)
print("GHSOM Model Persistence Example")
print("=" * 60)

# Generate data
print("\n1. Generating and preprocessing data...")
data = np.random.rand(200, 10)
print(f"   Dataset shape: {data.shape}")

# Training configuration
config = {
    "t1": 0.5,
    "t2": 0.05,
    "learning_rate": 0.1,
    "decay": 0.9,
    "gaussian_sigma": 1.0,
    "epochs": 50
}

# Train model
print("\n2. Training GHSOM model...")
ghsom = GHSOM(
    input_dataset=data,
    t1=config["t1"],
    t2=config["t2"],
    learning_rate=config["learning_rate"],
    decay=config["decay"],
    gaussian_sigma=config["gaussian_sigma"]
)

model = ghsom.train(epochs_number=config["epochs"], n_workers=-1)
print("   Training complete!")

# Calculate some stats before saving
def count_neurons(neuron):
    """Count total neurons in the hierarchy."""
    count = 1
    if neuron.child_map is not None:
        for child_neuron in neuron.child_map.neurons.values():
            count += count_neurons(child_neuron)
    return count

total_neurons = count_neurons(model)
root_shape = model.child_map.map_shape()

# Save model
print("\n3. Saving model...")
model_path = exp_dir / "ghsom_model.pkl"
save_model(model, str(model_path))
print(f"   Model saved to: {model_path}")

# Save metadata
print("\n4. Saving metadata...")
metadata = {
    "config": config,
    "dataset_shape": list(data.shape),
    "trained_at": datetime.now().isoformat(),
    "model_path": str(model_path),
    "total_neurons": total_neurons,
    "root_map_shape": list(root_shape)
}

metadata_path = exp_dir / "metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   Metadata saved to: {metadata_path}")

# Load metadata
print("\n5. Loading metadata...")
with open(metadata_path, 'r') as f:
    loaded_metadata = json.load(f)

print(f"   Model trained at: {loaded_metadata['trained_at']}")
print(f"   Configuration: t1={loaded_metadata['config']['t1']}, t2={loaded_metadata['config']['t2']}")
print(f"   Dataset shape: {tuple(loaded_metadata['dataset_shape'])}")

# Load model
print("\n6. Loading model...")
loaded_model = load_model(str(model_path))
print("   Model loaded successfully!")

# Verify loaded model
print("\n7. Verifying loaded model...")
loaded_root_shape = loaded_model.child_map.map_shape()
loaded_total_neurons = count_neurons(loaded_model)

print(f"   Original root map shape: {root_shape}")
print(f"   Loaded root map shape:   {loaded_root_shape}")
print(f"   Match: {root_shape == loaded_root_shape}")

print(f"\n   Original total neurons: {total_neurons}")
print(f"   Loaded total neurons:   {loaded_total_neurons}")
print(f"   Match: {total_neurons == loaded_total_neurons}")

# Test prediction with loaded model
print("\n8. Testing prediction with loaded model...")
test_sample = data[0]

def find_bmu(neuron, sample):
    """Find best matching unit for a sample."""
    while neuron.child_map is not None:
        gsom_map = neuron.child_map
        winner = gsom_map.winner_neuron(sample)[0][0]
        neuron = winner
    return neuron

bmu = find_bmu(loaded_model, test_sample)
print(f"   BMU position: {bmu.position}")
print(f"   Distance to sample: {bmu.activation(test_sample):.6f}")

print("\n" + "=" * 60)
print("Model Persistence Example Completed Successfully!")
print("=" * 60)
print(f"\nSaved files:")
print(f"  - {model_path}")
print(f"  - {metadata_path}")
