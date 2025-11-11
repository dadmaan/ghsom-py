# Contributing

Thank you for your interest in contributing to GHSOM-Py!

For detailed contribution guidelines, please see our [CONTRIBUTING.md](https://github.com/dadmaan/ghsom-py/blob/main/CONTRIBUTING.md) file in the repository.

## Quick Links

- [Code of Conduct](https://github.com/dadmaan/ghsom-py/blob/main/CONTRIBUTING.md#code-of-conduct)
- [Development Setup](https://github.com/dadmaan/ghsom-py/blob/main/CONTRIBUTING.md#development-setup)
- [Code Standards](https://github.com/dadmaan/ghsom-py/blob/main/CONTRIBUTING.md#code-standards)
- [Pull Request Process](https://github.com/dadmaan/ghsom-py/blob/main/CONTRIBUTING.md#pull-request-process)

## Quick Start for Contributors

1. Fork the repository on GitHub
2. Clone your fork locally
3. Install in development mode: `pip install -e .[dev]`
4. Create a branch: `git checkout -b feature/your-feature`
5. Make changes and add tests
6. Run tests: `pytest tests/`
7. Format code: `black ghsom/ tests/`
8. Submit a pull request

## Code Style

- **Formatting**: Black with line length 100
- **Linting**: Ruff
- **Type hints**: Required for all public APIs
- **Docstrings**: Google style

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=ghsom --cov-report=html

# Run specific test
pytest tests/test_ghsom_training.py -v
```

## Building Documentation

```bash
# Install docs dependencies
pip install .[docs]

# Serve docs locally
mkdocs serve

# View at http://127.0.0.1:8000
```

## Getting Help

- Open a [GitHub Issue](https://github.com/dadmaan/ghsom-py/issues) for bugs
- Start a [Discussion](https://github.com/dadmaan/ghsom-py/discussions) for questions
- Check our [documentation](https://dadmaan.github.io/ghsom-py/) for guides

We welcome contributions from everyone! 🎉
