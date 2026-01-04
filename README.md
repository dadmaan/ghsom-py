# GHSOM-Py

[![Tests](https://github.com/dadmaan/ghsom-py/actions/workflows/tests.yml/badge.svg)](https://github.com/dadmaan/ghsom-py/actions/workflows/tests.yml)
[![Code Quality](https://github.com/dadmaan/ghsom-py/actions/workflows/quality.yml/badge.svg)](https://github.com/dadmaan/ghsom-py/actions/workflows/quality.yml)
[![Python Version](https://img.shields.io/pypi/pyversions/ghsom-py)](https://pypi.org/project/ghsom-py/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A pure Python implementation of the **Growing Hierarchical Self-Organizing Map (GHSOM)** algorithm for unsupervised learning and hierarchical clustering.

## Features

- 🚀 **Pure Python**: Lightweight implementation with minimal dependencies (numpy only)
- 📊 **Hierarchical Clustering**: Automatically grows hierarchical map structure
- ⚡ **Parallel Training**: Multi-core support for faster training
- 🔌 **Extensible**: Callback system for tracking and custom integrations
- 📦 **Well-Tested**: Comprehensive test suite with >90% coverage
- 🎯 **Type-Safe**: Full type hints for better IDE support

## Installation

```bash
pip install ghsom-py
```

### Optional Dependencies

```bash
# For WandB tracking
pip install ghsom-py[wandb]

# For utility functions using pandas
pip install ghsom-py[utils]

# For development
pip install ghsom-py[dev]

# All optional dependencies
pip install ghsom-py[all]
```

## Quick Start

```python
import numpy as np
from ghsom import GHSOM

# Create sample data
data = np.random.rand(100, 10)

# Initialize GHSOM
ghsom = GHSOM(
    input_dataset=data,
    t1=0.5,              # Growth threshold
    t2=0.05,             # Stopping criterion
    learning_rate=0.1,
    decay=0.9,
    gaussian_sigma=1.0
)

# Train the model
model = ghsom.train(
    epochs_number=50,
    n_workers=-1  # Use all CPU cores
)

# Access the trained hierarchy
root_map = model
print(f"Root map shape: {root_map.map_shape()}")
```

## Using Callbacks

Track training progress with callbacks:

```python
from ghsom import GHSOM
from ghsom.callbacks import TrackingCallback

class MyCallback(TrackingCallback):
    def on_train_begin(self, config):
        print(f"Training started with config: {config}")

    def on_map_created(self, metrics):
        print(f"New map created: {metrics}")

    def on_train_end(self, results):
        print(f"Training complete: {results}")

# Use callback during training
ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
model = ghsom.train(epochs_number=50, callbacks=[MyCallback()])
```

### WandB Integration

```python
from ghsom.callbacks import WandBCallback

callback = WandBCallback(
    project="my-project",
    name="ghsom-experiment",
    config={"t1": 0.5, "t2": 0.05}
)

model = ghsom.train(epochs_number=50, callbacks=[callback])
```

## Save and Load Models

```python
from ghsom.io import save_model, load_model

# Save trained model
save_model(model, 'ghsom_model.pkl')

# Load model later
loaded_model = load_model('ghsom_model.pkl')
```

## Documentation

Full documentation is available at [https://dadmaan.github.io/ghsom-py/](https://dadmaan.github.io/ghsom-py/)

## Algorithm Overview

The Growing Hierarchical Self-Organizing Map (GHSOM) is an unsupervised neural network that:

1. **Grows dynamically** based on data characteristics
2. **Creates hierarchical structure** for multi-level clustering
3. **Adapts map size** automatically (no need to specify map dimensions)
4. **Preserves topology** of the input space

### Key Parameters

- **t1**: Growth threshold - controls when a neuron grows a child map
- **t2**: Global stopping criterion - controls overall hierarchy depth
- **learning_rate**: Initial learning rate for weight updates
- **gaussian_sigma**: Initial neighborhood radius
- **decay**: Decay rate for learning rate and sigma

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0

## Development

```bash
# Clone repository
git clone https://github.com/dadmaan/ghsom-py.git
cd ghsom-py

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/

# Format code
black ghsom/ tests/

# Lint
ruff check ghsom/ tests/

# Type check
mypy ghsom/
```

## Citation

If you use GHSOM-Py in your research, please cite the original GHSOM paper:

```bibtex
@inproceedings{rauber2002growing,
  title={The growing hierarchical self-organizing map: exploratory analysis of high-dimensional data},
  author={Rauber, Andreas and Merkl, Dieter and Dittenbach, Michael},
  booktitle={IEEE Transactions on Neural Networks},
  volume={13},
  number={6},
  pages={1331--1341},
  year={2002},
  publisher={IEEE}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Related Projects

- [ghsom-toolkits](https://github.com/dadmaan/ghsom-toolkits) - Visualization and analysis tools for GHSOM
- [aria](https://github.com/dadmaan/aria) - Multi-agent RL framework for user-centric music generation

## Acknowledgments

This implementation is based on the GHSOM algorithm by Rauber et al. (2002). It was originally forked from [enriciv][https://github.com/enricivi/growing_hierarchical_som].
