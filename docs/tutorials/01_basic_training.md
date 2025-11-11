# Tutorial 1: Training Your First GHSOM Model

This tutorial walks you through training a GHSOM model from scratch, explaining each step in detail.

## What You'll Learn

- How to prepare your data for GHSOM
- How to configure GHSOM parameters
- How to train a model
- How to interpret the results

## Prerequisites

```bash
pip install ghsom-py numpy
```

## Step 1: Import and Prepare Data

```python
import numpy as np
from ghsom import GHSOM

# Generate synthetic data for this tutorial
# In practice, you'd use your own dataset
np.random.seed(42)
n_samples = 200
n_features = 8

# Create clustered data
data = np.concatenate([
    np.random.randn(n_samples // 2, n_features) + [2, 2, 0, 0, 1, 1, 0, 0],
    np.random.randn(n_samples // 2, n_features) + [-2, -2, 1, 1, 0, 0, 1, 1]
])

print(f"Dataset shape: {data.shape}")
print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")
```

## Step 2: Data Preprocessing

While GHSOM can work with raw data, normalizing your features is recommended:

```python
from sklearn.preprocessing import StandardScaler

# Standardize features to have mean=0 and std=1
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

print(f"Normalized data range: [{data_normalized.min():.2f}, {data_normalized.max():.2f}]")
```

## Step 3: Configure GHSOM Parameters

```python
# Initialize GHSOM with parameters
ghsom = GHSOM(
    input_dataset=data_normalized,
    t1=0.5,              # Growth threshold
    t2=0.05,             # Stopping criterion
    learning_rate=0.1,   # Initial learning rate
    decay=0.9,           # Decay rate
    gaussian_sigma=1.0   # Initial neighborhood size
)
```

### Parameter Guidelines

**Growth Threshold (t1)**:
- Range: 0.0 to 1.0
- Smaller values → larger maps with more neurons
- Typical: 0.3 to 0.7
- Use 0.5 as a good starting point

**Stopping Criterion (t2)**:
- Range: 0.0 to 1.0
- Smaller values → deeper hierarchies
- Typical: 0.01 to 0.1
- Use 0.05 as a good starting point

**Learning Rate**:
- Range: 0.0 to 1.0
- Higher values → faster adaptation but less stability
- Typical: 0.05 to 0.3
- Use 0.1 as a good starting point

**Decay**:
- Range: 0.0 to 1.0
- Controls how quickly learning rate decreases
- Typical: 0.8 to 0.95
- Use 0.9 as a good starting point

**Gaussian Sigma**:
- Positive float
- Initial neighborhood radius
- Typical: 1.0 to 3.0
- Use 1.0 for small maps, 2.0-3.0 for larger maps

## Step 4: Train the Model

```python
# Train with default settings
model = ghsom.train(
    epochs_number=50,    # Number of training epochs
    n_workers=-1         # Use all CPU cores
)

print("Training complete!")
```

### Training Parameters

- **epochs_number**: How many times to iterate over the data (default: 15)
- **n_workers**: Number of parallel workers (-1 = all CPU cores, None = automatic)
- **dataset_percentage**: Fraction of data to use per training round (default: 0.25)
- **min_dataset_size**: Minimum dataset size for training (default: 1)
- **seed**: Random seed for reproducibility (default: None)

## Step 5: Explore the Results

```python
# Access the root neuron
root = model

# Access the root map
root_map = root.child_map

print(f"Root map shape: {root_map.map_shape()}")
print(f"Number of neurons in root map: {len(root_map.neurons)}")

# Explore the hierarchy
def count_neurons(neuron, level=0):
    """Recursively count neurons in the hierarchy."""
    count = 1
    if neuron.child_map is not None:
        for child_neuron in neuron.child_map.neurons.values():
            count += count_neurons(child_neuron, level + 1)
    return count

total_neurons = count_neurons(root)
print(f"Total neurons in hierarchy: {total_neurons}")

# Check which neurons have child maps
neurons_with_children = 0
for pos, neuron in root_map.neurons.items():
    if neuron.child_map is not None:
        neurons_with_children += 1
        print(f"Neuron at {pos} has a child map of shape {neuron.child_map.map_shape()}")

print(f"Neurons with child maps: {neurons_with_children}/{len(root_map.neurons)}")
```

## Step 6: Find Best Matching Units

```python
# Find the best matching unit for a sample
sample = data_normalized[0]

def find_bmu(neuron, data_point):
    """Find the best matching unit for a data point."""
    # Traverse down the hierarchy
    while neuron.child_map is not None:
        gsom_map = neuron.child_map
        winner = gsom_map.winner_neuron(data_point)[0][0]
        neuron = winner
    return neuron

bmu = find_bmu(root, sample)
print(f"Best matching unit position: {bmu.position}")
print(f"Distance to sample: {bmu.activation(sample):.4f}")
```

## Complete Example

Here's the complete code:

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from ghsom import GHSOM

# Prepare data
np.random.seed(42)
data = np.random.rand(200, 8)
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Initialize and train
ghsom = GHSOM(
    input_dataset=data_normalized,
    t1=0.5,
    t2=0.05,
    learning_rate=0.1,
    decay=0.9,
    gaussian_sigma=1.0
)

model = ghsom.train(epochs_number=50, n_workers=-1)

# Analyze results
print(f"Root map shape: {model.child_map.map_shape()}")
print(f"Training complete!")
```

## Next Steps

- [Tutorial 2: Using Callbacks](02_callbacks.md) - Track training progress
- [Tutorial 3: Model Persistence](03_persistence.md) - Save and load models
- [Tutorial 4: Model Evaluation](04_evaluation.md) - Evaluate model quality
