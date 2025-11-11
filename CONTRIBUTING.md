# Contributing to GHSOM-Py

Thank you for your interest in contributing to GHSOM-Py! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

We are committed to providing a welcoming and inclusive experience for everyone. We expect all contributors to:

- Be respectful and considerate
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Types of Contributions

We welcome many types of contributions:

- **Bug reports**: Report bugs through GitHub issues
- **Bug fixes**: Submit pull requests to fix bugs
- **Feature requests**: Suggest new features via issues
- **Feature implementations**: Implement new features
- **Documentation**: Improve or add documentation
- **Examples**: Add new examples or improve existing ones
- **Tests**: Add or improve test coverage
- **Performance**: Optimize existing code

### Finding Issues

Good first issues for new contributors:
- Look for issues tagged with `good first issue`
- Check issues tagged with `help wanted`
- Documentation improvements are always welcome

## Development Setup

### Prerequisites

- Python >= 3.8
- Git
- pip

### Setting Up Your Environment

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ghsom-py.git
   cd ghsom-py
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/dadmaan/ghsom-py.git
   ```

4. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install in development mode**:
   ```bash
   pip install -e .[dev]
   ```

6. **Install pre-commit hooks** (optional but recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Development Workflow

### Creating a Branch

Create a new branch for your work:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `test/` - Test additions or changes
- `refactor/` - Code refactoring

### Making Changes

1. **Write your code**
2. **Add tests** for new functionality
3. **Update documentation** if needed
4. **Run tests locally**:
   ```bash
   pytest tests/
   ```

5. **Check code quality**:
   ```bash
   # Format code
   black ghsom/ tests/

   # Lint code
   ruff check ghsom/ tests/

   # Type check
   mypy ghsom/
   ```

### Committing Changes

Write clear, concise commit messages:

```bash
git add .
git commit -m "Add feature X to improve Y

- Detailed description of changes
- Why the change was needed
- Any breaking changes"
```

Commit message guidelines:
- Use present tense ("Add feature" not "Added feature")
- First line should be 50 characters or less
- Add detailed description after a blank line if needed
- Reference issues: "Fixes #123" or "Relates to #456"

## Code Standards

### Style Guide

We follow these conventions:

- **Code formatting**: [Black](https://black.readthedocs.io/) with line length 100
- **Linting**: [Ruff](https://github.com/astral-sh/ruff)
- **Type hints**: Required for all public APIs
- **Docstrings**: Google style for all public functions/classes

### Code Formatting

Run Black before committing:

```bash
black ghsom/ tests/ --line-length 100
```

### Linting

Run Ruff to check for issues:

```bash
ruff check ghsom/ tests/
```

Fix auto-fixable issues:

```bash
ruff check ghsom/ tests/ --fix
```

### Type Hints

All public APIs must have type hints:

```python
def train(
    self,
    epochs_number: int = 15,
    dataset_percentage: float = 0.25,
    seed: Optional[int] = None
) -> Neuron:
    """Train the GHSOM model."""
    pass
```

Check types with mypy:

```bash
mypy ghsom/
```

### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """
    Brief description of the function.

    More detailed description if needed.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param2 is negative.

    Example:
        ```python
        result = example_function("test", 42)
        print(result)
        ```
    """
    pass
```

## Testing

### Running Tests

Run all tests:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest tests/ --cov=ghsom --cov-report=html
```

Run specific test:

```bash
pytest tests/test_ghsom_training.py::TestGHSOMTraining::test_basic_training
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names
- Test both success and failure cases
- Aim for >90% code coverage

Example test:

```python
def test_ghsom_initialization():
    """Test that GHSOM initializes correctly."""
    data = np.random.rand(100, 10)
    ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)

    assert ghsom is not None
    assert ghsom._GHSOM__t1 == 0.5
    assert ghsom._GHSOM__t2 == 0.05
```

### Test Coverage Requirements

- Minimum 90% coverage for new code
- Core modules should have >95% coverage
- All public APIs must be tested
- Edge cases should be covered

## Documentation

### Building Documentation Locally

1. Install documentation dependencies:
   ```bash
   pip install .[docs]
   ```

2. Build the documentation:
   ```bash
   cd ghsom-py
   mkdocs serve
   ```

3. View at `http://127.0.0.1:8000`

### Documentation Standards

- Update docstrings for changed functions
- Add examples for new features
- Update tutorials if behavior changes
- Keep API reference up to date

### Adding Examples

When adding examples:

1. Create a new file in `examples/`
2. Follow the naming pattern: `##_description.py`
3. Include docstring at the top
4. Add comments explaining key steps
5. Update `examples/README.md`

## Pull Request Process

### Before Submitting

Checklist before submitting a PR:

- [ ] Code follows style guidelines (black, ruff)
- [ ] All tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated if needed
- [ ] Type hints added for new code
- [ ] Commit messages are clear and descriptive

### Submitting a Pull Request

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub

3. **Fill out the PR template** with:
   - Description of changes
   - Related issue numbers
   - Type of change (bugfix, feature, docs, etc.)
   - Testing done
   - Checklist completion

4. **Wait for review**:
   - Address reviewer comments
   - Push additional commits if needed
   - Request re-review when ready

### PR Review Process

- At least one maintainer review required
- All CI checks must pass
- No unresolved comments
- Branch must be up to date with main

### After PR is Merged

1. **Delete your branch**:
   ```bash
   git checkout main
   git pull upstream main
   git branch -d feature/your-feature-name
   ```

2. **Update your fork**:
   ```bash
   git push origin main
   ```

## Release Process

(For maintainers)

### Version Numbers

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

### Creating a Release

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create a git tag:
   ```bash
   git tag -a v0.2.0 -m "Release version 0.2.0"
   git push upstream v0.2.0
   ```
4. GitHub Actions will automatically build and publish to PyPI

## Getting Help

- **Questions**: Open a discussion on GitHub
- **Bugs**: Report via GitHub issues
- **Security**: Email maintainers directly (see SECURITY.md)
- **Chat**: Join our community (link TBD)

## Recognition

Contributors will be:
- Listed in the project's contributors page
- Mentioned in release notes for significant contributions
- Credited in academic citations when appropriate

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to GHSOM-Py! 🎉
