"""Model persistence utilities for saving and loading GHSOM models."""

import pickle


def save_model(model, filename):
    """
    Save a trained GHSOM model to disk using pickle.

    Args:
        model: The trained GHSOM model (root Neuron) to save.
        filename (str): Path to the file where the model will be saved.

    Returns:
        None

    Example:
        ```python
        from ghsom import GHSOM
        from ghsom.io import save_model

        # Train a model
        ghsom = GHSOM(input_dataset=data, t1=0.5, t2=0.05)
        model = ghsom.train(epochs_number=50)

        # Save the trained model
        save_model(model, 'ghsom_model.pkl')
        ```
    """
    with open(filename, "wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(filename):
    """
    Load a trained GHSOM model from disk.

    Args:
        filename (str): Path to the saved model file.

    Returns:
        Neuron: The loaded GHSOM model (root Neuron).

    Example:
        ```python
        from ghsom.io import load_model
        from ghsom.evaluation.metrics import hierarchy_depth

        # Load a previously saved model
        model = load_model('ghsom_model.pkl')

        # Use the model for analysis
        depth = hierarchy_depth(model)
        print(f"Model hierarchy depth: {depth}")
        ```
    """
    with open(filename, "rb") as file:
        return pickle.load(file)
