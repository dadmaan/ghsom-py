# GHSOM-Py Quick Reference

## Installation

```bash
# From source (development)
cd /workspace/ghsom-py
pip install -e .

# With optional dependencies
pip install -e .[wandb]      # WandB tracking
pip install -e .[utils]      # Pandas utilities
pip install -e .[dev]        # Development tools
pip install -e .[all]        # Everything
```

## Basic Usage

```python
import numpy as np
from ghsom import GHSOM

# Create data
data = np.random.rand(100, 10)

# Train GHSOM
ghsom = GHSOM(
    input_dataset=data,
    t1=0.5,              # Growth threshold
    t2=0.05,             # Stopping criterion
    learning_rate=0.1,
    decay=0.9,
    gaussian_sigma=1.0
)

model = ghsom.train(epochs_number=50)
```

## Save/Load Models

```python
from ghsom.io import save_model, load_model

# Save
save_model(model, 'my_model.pkl')

# Load
loaded = load_model('my_model.pkl')
```

## Using Callbacks

```python
from ghsom.callbacks import TrackingCallback

class MyCallback(TrackingCallback):
    def on_train_begin(self, config):
        print("Training started!")

    def on_map_created(self, metrics):
        print(f"Map created: {metrics}")

    def on_train_end(self, results):
        print("Training complete!")

# Use it
model = ghsom.train(callbacks=[MyCallback()])
```

## WandB Integration

```python
from ghsom.callbacks import WandBCallback

callback = WandBCallback(
    project="my-project",
    name="experiment-1"
)

model = ghsom.train(callbacks=[callback])
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=ghsom --cov-report=term-missing

# Run specific test file
pytest tests/test_imports.py -v
```

## Building

```bash
# Build distribution
python -m build

# Check build
ls dist/
# ghsom_py-0.1.0-py3-none-any.whl
# ghsom_py-0.1.0.tar.gz
```

## Development

```bash
# Format code
black ghsom/ tests/ --line-length 100

# Lint
ruff check ghsom/ tests/

# Type check
mypy ghsom/ --ignore-missing-imports
```

## Package Structure

```
ghsom/
├── core/           # GHSOM, GSOM, Neuron
├── builders/       # NeuronBuilder
├── evaluation/     # Metrics
├── io/             # save_model, load_model, parsing
├── callbacks/      # TrackingCallback, WandBCallback
└── utils/          # Helper functions
```

## Public API

```python
# Core classes
from ghsom import GHSOM, GSOM, Neuron, NeuronBuilder

# Callbacks
from ghsom import TrackingCallback
from ghsom.callbacks import WandBCallback

# I/O
from ghsom.io import save_model, load_model

# Version
from ghsom import __version__
```

## Common Parameters

**GHSOM Parameters:**
- `input_dataset`: Training data (np.ndarray)
- `t1`: Growth threshold (smaller = larger maps)
- `t2`: Stopping criterion (global depth control)
- `learning_rate`: Initial learning rate
- `decay`: Decay rate for learning rate and sigma
- `gaussian_sigma`: Neighborhood width

**Train Parameters:**
- `epochs_number`: Number of training epochs (default: 15)
- `dataset_percentage`: Percentage of dataset per epoch (default: 0.25)
- `min_dataset_size`: Minimum dataset size (default: 1)
- `n_workers`: Number of parallel workers (-1 = all cores)
- `callbacks`: List of TrackingCallback instances

## Version Info

Current Version: **0.1.0**

Requirements:
- Python >= 3.8
- numpy >= 1.20.0

## Next Steps

See:
- Full documentation: `README.md`
- Version history: `CHANGELOG.md`
- Implementation details: `PHASE2_SUMMARY.md`
