"""Base callback interface for GHSOM training."""

from abc import ABC, abstractmethod


class TrackingCallback(ABC):
    """
    Abstract base class for GHSOM training callbacks.

    Callbacks provide hooks for monitoring and tracking the training process
    without coupling the core algorithm to specific tracking implementations.

    Subclass this to create custom callbacks for logging, visualization,
    or integration with tracking tools like WandB, MLflow, TensorBoard, etc.
    """

    @abstractmethod
    def on_train_begin(self, config):
        """
        Called at the beginning of training.

        :param config: dict
            Training configuration including hyperparameters:
            - learning_rate: Initial learning rate
            - gaussian_sigma: Initial Gaussian sigma
            - decay: Decay rate
            - t1: Growth threshold
            - t2: Global stopping criterion
            - grow_maxiter: Max iterations for growing
            - dataset_percentage: Percentage of dataset used
            - min_dataset_size: Minimum dataset size
            - epochs: Number of epochs
        """
        pass

    @abstractmethod
    def on_map_created(self, metrics):
        """
        Called when a new map is created during training.

        :param metrics: dict
            Metrics for the newly created map:
            - qe or mqe: Quantization error
            - neuron_position: Position of the neuron
        """
        pass

    @abstractmethod
    def on_train_end(self, results):
        """
        Called at the end of training.

        :param results: dict
            Final training results:
            - total_neurons: Total number of neurons created
        """
        pass
