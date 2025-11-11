# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-10-08

### Added
- Initial release of `ghsom-py` package
- Core GHSOM algorithm implementation
  - `GHSOM` class for hierarchical self-organizing maps
  - `GSOM` class for growing self-organizing maps
  - `Neuron` class for individual map units
- Neuron builder utilities for creating neurons
- Evaluation metrics for quantization error and mapping quality
- I/O utilities for model persistence
  - `save_model()` function for saving trained models
  - `load_model()` function for loading saved models
- Parsing utilities for GHSOM structures
- Helper utilities for hierarchy analysis and neuron operations
- Callback system for extensible training tracking
  - `TrackingCallback` abstract base class
  - `WandBCallback` for Weights & Biases integration
- Comprehensive test suite
  - Import tests
  - Basic functionality tests
  - 12 passing tests with 20% code coverage
- CI/CD workflows
  - Multi-platform testing (Ubuntu, macOS, Windows)
  - Multi-version Python support (3.8, 3.9, 3.10, 3.11)
  - Code quality checks (Black, Ruff, MyPy)
  - Coverage reporting with Codecov integration
- Complete documentation
  - README with installation and quick start
  - API documentation in docstrings
  - Usage examples
  - Citation information

### Dependencies
- Core: numpy >= 1.20.0
- Optional: wandb >= 0.12.0 (for WandB tracking)
- Optional: pandas >= 1.3.0 (for utility functions)
- Development: pytest, pytest-cov, black, ruff, mypy

### Package Structure
```
ghsom/
├── core/           # Core algorithm (GHSOM, GSOM, Neuron)
├── builders/       # Neuron builders
├── evaluation/     # Metrics and evaluation
├── io/             # Persistence and parsing
├── callbacks/      # Training callbacks
└── utils/          # Helper utilities
```

### Public API
- `ghsom.GHSOM` - Main GHSOM class
- `ghsom.GSOM` - Growing SOM class
- `ghsom.Neuron` - Neuron class
- `ghsom.NeuronBuilder` - Neuron factory
- `ghsom.TrackingCallback` - Callback base class
- `ghsom.io.save_model()` - Save model to file
- `ghsom.io.load_model()` - Load model from file

### Configuration
- Supports Python 3.8+
- Type hints for all public APIs
- Configurable via pyproject.toml
- MIT License

### Known Limitations
- Test coverage at 20% (comprehensive testing in Phase 3)
- Some utility functions require pandas (optional dependency)
- Visualization features deferred to `ghsom-toolkits` package

### Migration Notes
- Extracted from parent music generation RL project
- All imports changed from `src.models.ghsom.*` to `ghsom.*`
- Logging uses standard Python `logging` module
- WandB tracking moved to optional callback

---

## Development Notes

### Build Information
- Package can be built with `python -m build`
- Installable in development mode with `pip install -e .`
- Generates both wheel and source distribution

### Testing
Run tests with:
```bash
pytest tests/ -v
```

All 12 tests passing as of release 0.1.0.

### Future Plans
- Phase 3: Comprehensive testing (target >90% coverage)
- Phase 4: Complete documentation
- Phase 5: Visualization toolkit package (`ghsom-toolkits`)
- Phase 8: Integration with parent project

---

[0.1.0]: https://github.com/dadmaan/ghsom-py/releases/tag/v0.1.0
