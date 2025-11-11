# GHSOM-Py Examples

This directory contains runnable examples demonstrating various features of GHSOM-Py.

## Available Examples

### 1. Basic Training (`01_basic_training.py`)
**Difficulty:** Beginner
**Time:** ~2 minutes

Learn the basics of training a GHSOM model:
- Creating synthetic data
- Initializing GHSOM with parameters
- Training the model
- Analyzing the resulting hierarchy

```bash
python examples/01_basic_training.py
```

### 2. Using Callbacks (`02_with_callbacks.py`)
**Difficulty:** Beginner
**Time:** ~2 minutes

Monitor training progress with custom callbacks:
- Creating custom callback classes
- Tracking training metrics
- Logging quantization errors
- Timing training phases

```bash
python examples/02_with_callbacks.py
```

### 3. Saving and Loading Models (`03_save_and_load.py`)
**Difficulty:** Beginner
**Time:** ~2 minutes

Persist and reload trained models:
- Saving models to disk
- Storing training metadata
- Loading models for later use
- Verifying loaded models

```bash
python examples/03_save_and_load.py
```

### 4. Model Evaluation (`04_evaluation_metrics.py`)
**Difficulty:** Intermediate
**Time:** ~3 minutes

Evaluate model quality with various metrics:
- Built-in evaluation metrics
- Quantization error analysis
- Hierarchy structure analysis
- Neuron usage patterns

```bash
python examples/04_evaluation_metrics.py
```

### 5. Multiprocessing (`05_multiprocessing.py`)
**Difficulty:** Intermediate
**Time:** ~5 minutes

Optimize training performance:
- Parallel training with multiple workers
- Performance benchmarking
- Memory optimization techniques
- Best practices for different dataset sizes

```bash
python examples/05_multiprocessing.py
```

## Running All Examples

To run all examples sequentially:

```bash
cd ghsom-py
for example in examples/0*.py; do
    echo "Running $example..."
    python "$example"
    echo "---"
done
```

## Requirements

All examples require:
- Python >= 3.8
- ghsom-py installed
- NumPy

Install with:
```bash
pip install ghsom-py
```

## Generated Files

Some examples create output files:

- `03_save_and_load.py` creates: `experiment_output/`
- `05_multiprocessing.py` may create temporary files

These are safe to delete after running the examples.

## Modifying Examples

Feel free to modify these examples to experiment with:

- Different parameter values (t1, t2, learning_rate, etc.)
- Larger or smaller datasets
- Different data distributions
- Custom callbacks
- Alternative evaluation metrics

## Additional Resources

- [Documentation](https://ghsom-py.readthedocs.io)
- [Tutorial Series](../docs/tutorials/)
- [API Reference](../docs/api/)
- [Contributing Guide](../CONTRIBUTING.md)

## Getting Help

If you encounter issues:

1. Check the [documentation](https://ghsom-py.readthedocs.io)
2. Review the [tutorials](../docs/tutorials/)
3. Open an issue on [GitHub](https://github.com/dadmaan/ghsom-py/issues)

## License

These examples are part of GHSOM-Py and are licensed under the MIT License.
