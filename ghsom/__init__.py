"""
GHSOM-Py: Growing Hierarchical Self-Organizing Maps in Python
==============================================================

A pure Python implementation of the Growing Hierarchical Self-Organizing Map (GHSOM)
algorithm for unsupervised learning and hierarchical clustering.

Basic Usage
-----------
>>> import numpy as np
>>> from ghsom import GHSOM
>>>
>>> # Create sample data
>>> data = np.random.rand(100, 10)
>>>
>>> # Initialize and train GHSOM
>>> ghsom = GHSOM(
...     input_dataset=data,
...     t1=0.5,
...     t2=0.05,
...     learning_rate=0.1,
...     decay=0.9,
...     gaussian_sigma=1.0
... )
>>> model = ghsom.train(epochs_number=50)
"""

__version__ = "0.1.0"

from ghsom.core.ghsom import GHSOM
from ghsom.core.gsom import GSOM
from ghsom.core.neuron import Neuron
from ghsom.builders.neuron_builder import NeuronBuilder
from ghsom.callbacks.base import TrackingCallback

__all__ = [
    "GHSOM",
    "GSOM",
    "Neuron",
    "NeuronBuilder",
    "TrackingCallback",
    "__version__",
]
