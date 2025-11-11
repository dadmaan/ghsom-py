import logging
import os
from multiprocessing import Pool
from queue import Queue

import numpy as np

from ghsom.core.gsom import GSOM
from ghsom.builders.neuron_builder import NeuronBuilder


class GHSOM:
    def __init__(
        self,
        input_dataset,
        t1,
        t2,
        learning_rate,
        decay,
        gaussian_sigma,
        growing_metric="qe",
        logger=None,
    ):
        """
        Python based implementation of Growing Hierarchical Self-Organizing Maps.

        This implementation is based on the paper
        [The Growing Hierarchical Self-Organizing Map: Exploratory Analysis of High-Dimensional Data]
        (https://ieeexplore.ieee.org/document/1058070/)

        :param input_dataset: np.array, shape=(n_samples, n_features)
            The training data for the GHSOM.

        :param t1: float
            The growth rate of the SOM maps for layer i. When the map reaches a fraction of t1,
            the map should expand by creating new neurons to represent the data. The smaller
            the parameter is chosen the larger the resulting map will be, explaining its
            data at a higher granularity.

        :param t2: float
            The global stopping criterion, which directly influences the overall size of
            the resulting GHSOM. It controls the tradeoff between shallow or deep hierarchies.
            It can be an absolute value or a fraction specified by parameter.

        :param learning_rate: float
            The initial learning rate for the GHSOM.

        :param decay: float
            The decay rate for the learning rate and the Gaussian sigma.

        :param gaussian_sigma: float
            The initial width of the neighborhood function, specified in terms of the number of standard deviations.

        :param growing_metric: str, optional (default="qe")
            The metric used to evaluate the mapping quality of a SOM. Select "mqe" (Mean Quantization Error) or "qe" (Quantization Error).
            By default, the quantization error (QE) is used.

        :param logger: logging.Logger, optional (default=None)
            Custom logger instance. If not provided, uses standard logging.
        """
        self.__input_dataset = input_dataset
        self.__input_dimension = input_dataset.shape[1]

        self.__gaussian_sigma = gaussian_sigma
        self.__decay = decay
        self.__learning_rate = learning_rate
        self.__growing_metric = growing_metric
        self.__t1 = t1
        self.__t2 = t2

        self.__neuron_builder = NeuronBuilder(self.__t2, growing_metric)
        self.logger = logger or logging.getLogger(__name__)

    def train(
        self,
        epochs_number=15,
        dataset_percentage=0.25,
        min_dataset_size=1,
        seed=None,
        grow_maxiter=100,
        callbacks=None,
        n_workers=None,
    ):
        """
        Train the GHSOM model.
        Create an initial GSOM for the zero unit and then
        iteratively train GSOMs for all neurons that need a child map.
        It implements a breadth-first search algorithm to train a hierarchy
        of GSOM maps. The hierarchy is constructed in a top-down manner,
        starting from the root (the zero unit) and going down to
        the leaves (the neurons in the lowest level of the hierarchy).
        The training of the GSOM maps is parallelized using a process pool.

        :param epochs_number: int, optional (default=15)
            The number of training epochs.

        :param dataset_percentage: float, optional (default=0.25)
            The percentage of the dataset used for training.

        :param min_dataset_size: int, optional (default=1)
            The minimum size of the dataset used for training.

        :param seed: int, optional
            The seed for the random number generator.

        :param grow_maxiter: int, optional (default=100)
            The maximum number of iterations for growing the map.

        :param callbacks: list of TrackingCallback, optional (default=None)
            List of callback objects for monitoring training progress.
            Use WandBCallback for WandB tracking, or implement custom callbacks.

        :param n_workers: int, optional (default=None)
            The maximum number of worker processes to run on CPU cores.
            Pass "-1" to utilize all CPU cores, and "None" to determine automatically.

        :return: Neuron
            The root of the neuron tree.
        """
        callbacks = callbacks or []

        # Invoke on_train_begin callback
        config = {
            "learning_rate": self.__learning_rate,
            "gaussian_sigma": self.__gaussian_sigma,
            "decay": self.__decay,
            "t1": self.__t1,
            "t2": self.__t2,
            "grow_maxiter": grow_maxiter,
            "dataset_percentage": dataset_percentage,
            "min_dataset_size": min_dataset_size,
            "epochs": epochs_number,
        }
        for callback in callbacks:
            callback.on_train_begin(config)

        # Create the zero unit and its initial GSOM
        zero_unit = self.__init_zero_unit(seed=seed)

        # Create a queue for neurons that need to be trained
        neuron_queue = Queue()
        neuron_queue.put(zero_unit)

        # Create a process pool for parallel training
        n_workers = (
            os.cpu_count() if n_workers == -1 else n_workers
        )  # get maximum workers
        pool = Pool(processes=n_workers)

        # Keep track of the number of active data points
        active_dataset = len(zero_unit.input_dataset)

        # While there are neurons that need to be trained
        while not neuron_queue.empty():
            size = min(neuron_queue.qsize(), pool._processes)
            gmaps = dict()

            # For each neuron that needs to be trained
            for _ in range(size):
                neuron = neuron_queue.get()

                # Train the child map of the neuron in a separate process
                gmaps[neuron] = pool.apply_async(
                    neuron.child_map.train,
                    (
                        epochs_number,
                        self.__gaussian_sigma,
                        self.__learning_rate,
                        self.__decay,
                        dataset_percentage,
                        min_dataset_size,
                        seed,
                        grow_maxiter,
                    ),
                )

                active_dataset -= len(neuron.input_dataset)

            # Wait for the training to finish and
            # then update the child map of each neuron
            for neuron in gmaps:
                gmap = gmaps[neuron].get()
                neuron.child_map = gmap

                # If a neuron needs a child map, create a new GSOM for it
                neurons_to_expand = filter(
                    lambda _neuron: _neuron.needs_child_map(), gmap.neurons.values()
                )
                for _neuron in neurons_to_expand:
                    _neuron.child_map = self.__build_new_GSOM(
                        _neuron.compute_quantization_error(),
                        _neuron.input_dataset,
                        self.__new_map_weights(_neuron.position, gmap.weights_map[0]),
                    )

                    # Invoke on_map_created callback
                    metrics = {
                        self.__growing_metric: _neuron.compute_quantization_error(),
                        "neuron_position": str(_neuron.position),
                    }
                    for callback in callbacks:
                        callback.on_map_created(metrics)

                    # Add the neuron to the queue
                    neuron_queue.put(_neuron)

                    active_dataset += len(_neuron.input_dataset)

        # Return the zero unit, which is the root of the hierarchy
        pool.close()
        pool.join()

        # Invoke on_train_end callback
        results = {"total_neurons": active_dataset}
        for callback in callbacks:
            callback.on_train_end(results)

        return zero_unit

    def __init_zero_unit(self, seed):
        """
        Initialize the zero unit and create its initial GSOM.

        :param seed: int
            The seed for the random number generator.

        :return: Neuron
            The initialized zero unit.
        """
        zero_unit = self.__neuron_builder.zero_neuron(self.__input_dataset)

        zero_unit.child_map = self.__build_new_GSOM(
            self.__neuron_builder.zero_quantization_error,
            zero_unit.input_dataset,
            self.__calc_initial_random_weights(seed=seed),
        )

        return zero_unit

    # noinspection PyPep8Naming
    def __build_new_GSOM(self, parent_quantization_error, parent_dataset, weights_map):
        """
        Build a new GSOM with the given parent quantization error,
        parent dataset, and weights map.

        :param parent_quantization_error: float
            The quantization error of the parent neuron.

        :param parent_dataset: np.array, shape=(n_samples, n_features)
            The dataset of the parent neuron.

        :param weights_map: np.array, shape=(n_rows, n_columns, n_features)
            The weight map for the new GSOM.

        :return: GSOM
            The newly built GSOM.
        """
        return GSOM(
            (2, 2),
            parent_quantization_error,
            self.__t1,
            self.__input_dimension,
            weights_map,
            parent_dataset,
            self.__neuron_builder,
        )

    def __new_map_weights(self, parent_position, weights_map):
        """
        Calculate the weights for a new GSOM map.
        Generate a stencil around the parent position
        and average the weights of the neurons in the stencil
        to create the weights for the new map. It generates a stencil
        (a 3x3 grid of cells centered around a given cell) based on
        the parent neuron's position. It then averages the weights
        of the neurons in the stencil to create the weights for the new GSOM.
         ______ ______ ______
        |      |      |      |         child (2x2)
        | pnfp |      |      |          ______ ______
        |______|______|______|         |      |      |
        |      |      |      |         |(0,0) |(0,1) |
        |      |parent|      |  ---->  |______|______|
        |______|______|______|         |      |      |
        |      |      |      |         |(1,0) |(1,1) |
        |      |      |      |         |______|______|
        |______|______|______|

        :param parent_position: tuple
            The position of the parent neuron.

        :param weights_map: np.array, shape=(n_rows, n_columns, n_features)
            The existing weights map.

        :return: np.array, shape=(n_rows, n_columns, n_features)
            The weights for the new GSOM map.
        """

        child_weights = np.zeros(shape=(2, 2, self.__input_dimension))
        stencil = self.__generate_kernel_stencil(parent_position)
        for child_position in np.ndindex(2, 2):
            child_position = np.asarray(child_position)
            mask = self.__filter_out_of_bound_positions(
                child_position, stencil, weights_map.shape
            )

            weight = np.mean(
                self.__elements_from_positions_list(weights_map, mask), axis=0
            )

            child_weights[child_position] = weight

        return child_weights

    @staticmethod
    def __elements_from_positions_list(matrix, positions_list):
        """
        Get the elements from a matrix at a list of positions.

        :param matrix: np.array
            The input matrix.

        :param positions_list: list of tuples
            The list of positions.

        :return: np.array
            The elements from the matrix at the specified positions.
        """
        return matrix[positions_list[:, 0], positions_list[:, 1]]

    def __filter_out_of_bound_positions(self, child_position, stencil, map_shape):
        """
        Filter out positions that are out of bounds of a given map shape.

        :param child_position: tuple
            The position of the child neuron.

        :param stencil: np.array
            The stencil array.

        :param map_shape: tuple
            The shape of the map.

        :return: np.array
            The filtered positions.
        """
        return np.asarray(
            list(
                filter(
                    lambda pos: self.__check_position(pos, map_shape),
                    stencil + child_position,
                )
            )
        )

    def __calc_initial_random_weights(self, seed):
        """
        Calculate the initial random weights for a GSOM.
        Create a random weights map by selecting random data items from the input dataset.

        :param seed: int
            The seed for the random number generator.

        :return: np.array, shape=(n_rows, n_columns, n_features)
            The initial random weights.
        """
        random_generator = np.random.RandomState(seed)
        random_weights = np.zeros(shape=(2, 2, self.__input_dimension))
        for position in np.ndindex(2, 2):
            random_data_item = self.__input_dataset[
                random_generator.randint(len(self.__input_dataset))
            ]
            random_weights[position] = random_data_item

        return random_weights

    @staticmethod
    def __generate_kernel_stencil(parent_position):
        """
        Generate a stencil around a given position.
        The stencil is a list of positions that are adjacent to the given position.

        :param parent_position: tuple
            The position of the parent neuron.

        :return: np.array
            The generated kernel stencil.
        """
        row, col = parent_position
        return np.asarray(
            [(r, c) for r in range(row - 1, row + 1) for c in range(col - 1, col + 1)]
        )

    @staticmethod
    def __check_position(position, map_shape):
        """
        Check whether a position is within the bounds of a given map shape.

        :param position: tuple
            The position to check.

        :param map_shape: tuple
            The shape of the map.

        :return: bool
            True if the position is within the bounds of the map, False otherwise.
        """
        row, col = position
        map_rows, map_cols = map_shape[0], map_shape[1]
        return (row >= 0 and col >= 0) and (row < map_rows and col < map_cols)
