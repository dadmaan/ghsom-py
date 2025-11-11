# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-10-08

### Added
- Initial release of GHSOM-Py
- Core GHSOM implementation with hierarchical self-organizing maps
- GSOM (Growing Self-Organizing Map) base implementation
- Neuron and NeuronBuilder classes
- Callback system for training monitoring
- WandB integration via WandBCallback
- Model persistence (save/load) functionality
- Evaluation metrics:
  - Mean data centroid activation
  - Hierarchy depth calculation
  - Total neuron count
  - Neuron usage analysis
- Parallel training support with multiprocessing
- Comprehensive test suite (>90% coverage)
- Full type hints for better IDE support
- Documentation with MkDocs and Material theme

### Features
- Dynamic map growth based on data characteristics
- Automatic hierarchy creation
- Configurable growth thresholds (t1, t2)
- Multiple growing metrics (QE, MQE)
- Dataset sampling for memory efficiency
- Reproducible training with seed support

[0.1.0]: https://github.com/dadmaan/ghsom-py/releases/tag/v0.1.0
