# Installation

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0

## Basic Installation

Install GHSOM-Py using pip:

```bash
pip install ghsom-py
```

This installs the core package with minimal dependencies (numpy only).

## Optional Dependencies

### WandB Integration

For experiment tracking with Weights & Biases:

```bash
pip install ghsom-py[wandb]
```

### Utilities

For additional utility functions using pandas:

```bash
pip install ghsom-py[utils]
```

### Development Tools

For contributing to the project:

```bash
pip install ghsom-py[dev]
```

This includes:
- pytest and pytest-cov (testing)
- black (code formatting)
- ruff (linting)
- mypy (type checking)

### Documentation Tools

To build documentation locally:

```bash
pip install ghsom-py[docs]
```

### All Dependencies

To install everything:

```bash
pip install ghsom-py[all]
```

## Development Installation

To install from source for development:

```bash
# Clone the repository
git clone https://github.com/dadmaan/ghsom-py.git
cd ghsom-py

# Install in editable mode with dev dependencies
pip install -e .[dev]
```

## Verifying Installation

Verify your installation:

```python
import ghsom
print(ghsom.__version__)

from ghsom import GHSOM
import numpy as np

# Create a small test dataset
data = np.random.rand(10, 5)

# Initialize GHSOM
model = GHSOM(
    input_dataset=data,
    t1=0.5,
    t2=0.05,
    learning_rate=0.1,
    decay=0.9,
    gaussian_sigma=1.0
)

print("GHSOM-Py installed successfully!")
```

## Troubleshooting

### Import Errors

If you encounter import errors:

```bash
# Upgrade pip and reinstall
pip install --upgrade pip
pip install --force-reinstall ghsom-py
```

### NumPy Compatibility

If you have NumPy version conflicts:

```bash
# Install with specific NumPy version
pip install ghsom-py numpy==1.24.0
```

### Multiple Python Versions

If you have multiple Python versions:

```bash
# Use python3 explicitly
python3 -m pip install ghsom-py

# Or use a specific version
python3.10 -m pip install ghsom-py
```

## Next Steps

- [Quick Start Tutorial](quickstart.md) - Train your first model
- [API Reference](../api/core/ghsom.md) - Explore the API
