# Tutorial 3: Saving and Loading Models

Learn how to save trained GHSOM models and load them for later use.

## What You'll Learn

- How to save trained models
- How to load saved models
- Best practices for model persistence
- Handling different Python versions

## Why Save Models?

Training a GHSOM model can be time-consuming, especially with large datasets. Saving models allows you to:

- Reuse trained models without retraining
- Share models with collaborators
- Deploy models to production
- Archive experiment results

## Basic Save and Load

```python
import numpy as np
from ghsom import GHSOM
from ghsom.io import save_model, load_model

# Train a model
data = np.random.rand(200, 10)
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
model = ghsom.train(epochs_number=50)

# Save the model
save_model(model, 'ghsom_model.pkl')
print("Model saved to ghsom_model.pkl")

# Load the model later
loaded_model = load_model('ghsom_model.pkl')
print("Model loaded successfully")

# Verify it's the same
print(f"Original root map shape: {model.child_map.map_shape()}")
print(f"Loaded root map shape: {loaded_model.child_map.map_shape()}")
```

## Save with Metadata

It's good practice to save metadata alongside your model:

```python
import json
from datetime import datetime
from ghsom import GHSOM
from ghsom.io import save_model

# Train configuration
config = {
    "t1": 0.5,
    "t2": 0.05,
    "learning_rate": 0.1,
    "decay": 0.9,
    "gaussian_sigma": 1.0,
    "epochs": 50
}

# Train model
data = np.random.rand(200, 10)
ghsom = GHSOM(
    input_dataset=data,
    t1=config["t1"],
    t2=config["t2"],
    learning_rate=config["learning_rate"],
    decay=config["decay"],
    gaussian_sigma=config["gaussian_sigma"]
)
model = ghsom.train(epochs_number=config["epochs"])

# Save model
model_path = 'models/ghsom_model.pkl'
save_model(model, model_path)

# Save metadata
metadata = {
    "config": config,
    "dataset_shape": data.shape,
    "trained_at": datetime.now().isoformat(),
    "model_path": model_path
}

with open('models/ghsom_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("Model and metadata saved")
```

## Loading with Metadata

```python
import json
from ghsom.io import load_model

# Load metadata first
with open('models/ghsom_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Model trained at: {metadata['trained_at']}")
print(f"Configuration: {metadata['config']}")
print(f"Dataset shape: {metadata['dataset_shape']}")

# Load the model
model = load_model(metadata['model_path'])
print("Model loaded successfully")
```

## Organizing Multiple Models

```python
import os
from pathlib import Path
from ghsom import GHSOM
from ghsom.io import save_model

def save_experiment(model, config, experiment_name, base_dir='experiments'):
    """Save a model with organized structure."""
    # Create experiment directory
    exp_dir = Path(base_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = exp_dir / 'model.pkl'
    save_model(model, str(model_path))

    # Save config
    config_path = exp_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Experiment saved to {exp_dir}")
    return str(exp_dir)

# Usage
data = np.random.rand(200, 10)
config = {"t1": 0.5, "t2": 0.05, "epochs": 50}

ghsom = GHSOM(input_dataset=data, t1=config["t1"], t2=config["t2"])
model = ghsom.train(epochs_number=config["epochs"])

save_experiment(model, config, "experiment_001")
```

## Loading from Experiments

```python
from pathlib import Path
from ghsom.io import load_model
import json

def load_experiment(experiment_name, base_dir='experiments'):
    """Load a model from experiment directory."""
    exp_dir = Path(base_dir) / experiment_name

    # Load config
    config_path = exp_dir / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load model
    model_path = exp_dir / 'model.pkl'
    model = load_model(str(model_path))

    return model, config

# Usage
model, config = load_experiment("experiment_001")
print(f"Loaded model with config: {config}")
```

## Handling Large Models

For very large models, consider compression:

```python
import gzip
import pickle

def save_compressed_model(model, filename):
    """Save a model with gzip compression."""
    with gzip.open(filename + '.gz', 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_compressed_model(filename):
    """Load a compressed model."""
    with gzip.open(filename + '.gz', 'rb') as f:
        return pickle.load(f)

# Usage
data = np.random.rand(500, 20)
ghsom = GHSOM(input_dataset=data, t1=0.4, t2=0.05)
model = ghsom.train(epochs_number=50)

# Save compressed
save_compressed_model(model, 'large_model.pkl')

# Load compressed
model = load_compressed_model('large_model.pkl')
```

## Version Compatibility

When sharing models, document the versions:

```python
import ghsom
import numpy
import sys

def get_environment_info():
    """Get version information."""
    return {
        "ghsom_version": ghsom.__version__,
        "numpy_version": numpy.__version__,
        "python_version": sys.version
    }

# Save with environment info
metadata = {
    "config": config,
    "environment": get_environment_info()
}

with open('models/environment.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

## Best Practices

### 1. Use Descriptive Filenames

```python
from datetime import datetime

# Bad
save_model(model, "model.pkl")

# Good
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_model(model, f"ghsom_t1-0.5_t2-0.05_{timestamp}.pkl")
```

### 2. Validate After Loading

```python
from ghsom.io import load_model

def validate_model(model):
    """Validate a loaded model."""
    assert model is not None, "Model is None"
    assert hasattr(model, 'child_map'), "Model missing child_map"
    assert model.child_map is not None, "Root map is None"
    print("✓ Model validation passed")

# Load and validate
model = load_model('ghsom_model.pkl')
validate_model(model)
```

### 3. Clean Up Training Data

If you don't need the training data stored in neurons:

```python
def remove_training_data(neuron):
    """Recursively remove training data from model to reduce size."""
    neuron.input_dataset = None

    if neuron.child_map is not None:
        for child in neuron.child_map.neurons.values():
            remove_training_data(child)

# Usage
model = ghsom.train(epochs_number=50)
remove_training_data(model)
save_model(model, 'model_no_data.pkl')
```

**Warning**: After removing training data, you can't use functions that require it!

### 4. Backup Important Models

```python
import shutil
from datetime import datetime

def backup_model(model_path):
    """Create a timestamped backup."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{model_path}.backup_{timestamp}"
    shutil.copy(model_path, backup_path)
    print(f"Backup created: {backup_path}")

# Usage
save_model(model, 'important_model.pkl')
backup_model('important_model.pkl')
```

## Complete Example

```python
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from ghsom import GHSOM
from ghsom.io import save_model, load_model

# Configuration
config = {
    "t1": 0.5,
    "t2": 0.05,
    "learning_rate": 0.1,
    "decay": 0.9,
    "gaussian_sigma": 1.0,
    "epochs": 50
}

# Create experiment directory
exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
exp_dir = Path("experiments") / exp_name
exp_dir.mkdir(parents=True, exist_ok=True)

# Train model
data = np.random.rand(200, 10)
ghsom = GHSOM(
    input_dataset=data,
    t1=config["t1"],
    t2=config["t2"],
    learning_rate=config["learning_rate"],
    decay=config["decay"],
    gaussian_sigma=config["gaussian_sigma"]
)
model = ghsom.train(epochs_number=config["epochs"])

# Save everything
save_model(model, str(exp_dir / "model.pkl"))

metadata = {
    "config": config,
    "dataset_shape": list(data.shape),
    "timestamp": datetime.now().isoformat()
}

with open(exp_dir / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Experiment saved to {exp_dir}")

# Later: Load the model
loaded_model = load_model(str(exp_dir / "model.pkl"))
with open(exp_dir / "metadata.json", 'r') as f:
    loaded_metadata = json.load(f)

print(f"Loaded model from {loaded_metadata['timestamp']}")
```

## Next Steps

- [Tutorial 4: Model Evaluation](04_evaluation.md) - Evaluate saved models
- [Tutorial 5: Advanced Usage](05_advanced.md) - Production deployment
- [API Reference: IO](../api/io.md) - Complete IO API
