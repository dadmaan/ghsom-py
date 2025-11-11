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

## Quick Example

```python
import numpy as np
from ghsom import GHSOM

# Create sample data
data = np.random.rand(100, 10)

# Initialize and train GHSOM
ghsom = GHSOM(
    input_dataset=data,
    t1=0.5,              # Growth threshold
    t2=0.05,             # Stopping criterion
    learning_rate=0.1,
    decay=0.9,
    gaussian_sigma=1.0
)

# Train the model
model = ghsom.train(epochs_number=50, n_workers=-1)

# Access the trained hierarchy
print(f"Root map shape: {model.child_map.map_shape()}")
```

## What is GHSOM?

The Growing Hierarchical Self-Organizing Map (GHSOM) is an unsupervised neural network that:

1. **Grows dynamically** based on data characteristics
2. **Creates hierarchical structure** for multi-level clustering
3. **Adapts map size** automatically (no need to specify map dimensions)
4. **Preserves topology** of the input space

Unlike traditional Self-Organizing Maps (SOMs), GHSOM automatically determines the optimal map size and creates a hierarchy of maps to represent data at different levels of granularity.

## Key Parameters

- **t1**: Growth threshold - controls when a neuron grows a child map (smaller = larger maps)
- **t2**: Global stopping criterion - controls overall hierarchy depth (smaller = deeper hierarchies)
- **learning_rate**: Initial learning rate for weight updates (typical: 0.05-0.3)
- **gaussian_sigma**: Initial neighborhood radius (typical: 1.0-3.0)
- **decay**: Decay rate for learning rate and sigma (typical: 0.8-0.95)

## Installation

```bash
pip install ghsom-py
```

For additional features:

```bash
# With WandB tracking
pip install ghsom-py[wandb]

# For development
pip install ghsom-py[dev]

# All dependencies
pip install ghsom-py[all]
```

## Next Steps

- [Installation Guide](getting-started/installation.md) - Detailed installation instructions
- [Quick Start Tutorial](getting-started/quickstart.md) - Your first GHSOM model in 5 minutes
- [API Reference](api/core/ghsom.md) - Complete API documentation
- [Examples](https://github.com/dadmaan/ghsom-py/tree/main/examples) - Runnable code examples

## Citation

If you use GHSOM-Py in your research, please cite the original GHSOM paper:

```bibtex
@article{rauber2002growing,
  title={The growing hierarchical self-organizing map: exploratory analysis of high-dimensional data},
  author={Rauber, Andreas and Merkl, Dieter and Dittenbach, Michael},
  journal={IEEE Transactions on Neural Networks},
  volume={13},
  number={6},
  pages={1331--1341},
  year={2002},
  publisher={IEEE}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/dadmaan/ghsom-py/blob/main/LICENSE) file for details.
