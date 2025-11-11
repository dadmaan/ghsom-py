import logging

import numpy as np

logger = logging.getLogger(__name__)


def mean_data_centroid_activation(ghsom, dataset):
    """
    Calculates the mean and standard deviation of the activation values
    of the centroid neurons in a GHSOM for a given dataset.

    Args:
        ghsom (GHSOM): The GHSOM object.
        dataset (ndarray): The dataset for which centroid activations are calculated.

    Returns:
        float: Mean activation value of the centroid neurons.
        float: Standard deviation of the activation values of the centroid neurons.

    """

    distances = list()

    # Iterate over each data point in the dataset
    for data in dataset:
        _neuron = ghsom

        # Traverse down the GHSOM hierarchy to find the winning neuron for the data point
        while _neuron.child_map is not None:
            _gsom = _neuron.child_map
            _neuron = _gsom.winner_neuron(data)[0][0]

        # Calculate the activation value of the centroid neuron and add it to the distances list
        distances.append(_neuron.activation(data))

    # Convert the distances list to a NumPy array
    distances = np.asarray(a=distances, dtype=np.float32)

    # Calculate the mean and standard deviation of the activation values
    mean_activation = distances.mean()
    std_activation = distances.std()

    logger.info(f"Mean Activation Value of the Centroid Neurons: {mean_activation}")
    logger.info(
        f"Standard Deviation of Activation Values of Centroid Neurons: {std_activation}"
    )

    return mean_activation, std_activation


def __number_of_neurons(root):
    """
    Recursively calculates the total number of neurons in a hierarchical GSOM.

    Args:
        root (GSOM): The root GSOM map.

    Returns:
        int: The total number of neurons in the GSOM hierarchy.

    """

    # Get the dimensions of the root GSOM
    r, c = root.child_map.weights_map[0].shape[0:2]

    # Calculate the initial number of neurons in the root GSOM
    total_neurons = r * c

    # Recursively calculate the total number of neurons in child GSOMs
    for neuron in root.child_map.neurons.values():
        if neuron.child_map is not None:
            total_neurons += __number_of_neurons(neuron)

    # Return the total number of neurons
    return total_neurons


def get_total_number_of_neurons(root):
    """Get total number of neurons in GHSOM hierarchy."""
    total_num_neurons = __number_of_neurons(root)
    logger.info(f"Total Number of Neurons: {total_num_neurons}")
    return total_num_neurons


def dispersion_rate(ghsom, dataset):
    """
    Calculates the dispersion rate of a GHSOM for a given dataset.
    The dispersion rate is the ratio of the total number of neurons
    in the GHSOM to the number of used neurons, where used neurons
    are the unique neurons activated by the dataset.

    Args:
        ghsom (GHSOM): The GHSOM object.
        dataset (ndarray): The dataset used for calculating the dispersion rate.

    Returns:
        float: The dispersion rate of the GHSOM.

    """

    used_neurons = dict()

    # Iterate over each data point in the dataset
    for data in dataset:
        gsom_reference = ""
        neuron_reference = ""
        _neuron = ghsom

        # Traverse down the GHSOM hierarchy to find the winning neuron for the data point
        while _neuron.child_map is not None:
            _gsom = _neuron.child_map
            _neuron = _gsom.winner_neuron(data)[0][0]

            gsom_reference = str(_gsom)
            neuron_reference = str(_neuron)

        # Record the activated neuron with its references
        used_neurons[
            "{}-{}-{}".format(gsom_reference, neuron_reference, _neuron.position)
        ] = True

    # Calculate the number of used neurons
    used_neurons = len(used_neurons)

    # Calculate the dispersion rate by dividing the total number of neurons by the number of used neurons
    disp_rate = __number_of_neurons(ghsom) / used_neurons
    logger.info(f"Dispersion Rate: {disp_rate}")

    return disp_rate


def _get_number_of_clusters(map):
    """
    Recursively counts the number of leaf neurons in a hierarchical neural network.

    Parameters:
    - map (MapNode): The root map of the hierarchical neural network.

    Returns:
    - int: The total number of leaf neurons in the hierarchy.
    """
    count = 0
    for neuron in map.neurons.values():
        if neuron.child_map is not None:
            count += _get_number_of_clusters(neuron.child_map)
        else:
            count += 1

    return count


def get_number_of_clusters(root_map):
    """Get number of clusters (leaf neurons) in GHSOM."""
    count = _get_number_of_clusters(root_map)
    logger.info(f"Number of Clusters: {count}")
    return count


def _get_number_of_maps(root):
    """
    Recursively counts the number of maps (sub-networks) in the hierarchical neural network.

    Parameters:
    - root (MapNode): The root neuron of the hierarchical neural network.

    Returns:
    - int: The total number of maps in the hierarchy.
    """
    count = 0
    if root.child_map is not None:
        count = 1
        for unit in root.child_map.neurons.values():
            count += _get_number_of_maps(unit)

    return count


def get_number_of_maps(root):
    """Get number of maps in GHSOM hierarchy."""
    count = _get_number_of_maps(root)
    logger.info(f"Number of Maps: {count}")
    return count


def _get_ghsom_depth(map, level):
    """
    Calculates the depth of the Growing Hierarchical Self-Organizing Map (GHSOM) starting from a given map.

    Parameters:
    - map (MapNode): The starting map for depth calculation.
    - level (int): The current level (depth) of the map.

    Returns:
    - int: The maximum depth of the GHSOM hierarchy.
    """
    depths = [level]
    for neuron in map.neurons.values():
        if neuron.child_map is not None:
            depths.append(_get_ghsom_depth(neuron.child_map, level + 1))

    return max(depths)


def get_ghsom_depth(map, level):
    """Get depth of GHSOM hierarchy."""
    max_depth = _get_ghsom_depth(map, level)
    logger.info(f"Depth of GHSOM Hierarchy: {max_depth}")
    return max_depth


def _get_max_neurons_in_child_map(map):
    """
    Recursively determines the maximum number of neurons in any child map within the hierarchy.

    Parameters:
    - map (MapNode): The root neuron of the hierarchical neural network.

    Returns:
    - int: The maximum number of neurons in any child map.
    """
    sizes = [len(map.neurons)]
    for neuron in map.neurons.values():
        if neuron.child_map is not None:
            sizes.append(_get_max_neurons_in_child_map(neuron.child_map))

    return max(sizes)


def get_max_neurons_in_child_map(map):
    """Get maximum neurons in any child map."""
    max_size = _get_max_neurons_in_child_map(map)
    logger.info(f"Maximum Neurons of a Child Map in GHSOM: {max_size}")
    return max_size


def get_child_map_sizes(map):
    """
    Recursively retrieves the sizes (shape) of all child maps within the hierarchical neural network.

    Parameters:
    - map (MapNode): The root neuron of the hierarchical neural network.

    Returns:
    - list: A list of tuples representing the sizes (shapes) of all child maps in the hierarchy.
    """
    s = [map.weights_map[0].shape]
    for neuron in map.neurons.values():
        if neuron.child_map is not None:
            s.extend(get_child_map_sizes(neuron.child_map))
    return s


def get_mean_child_map_size(map):
    """
    Calculates the mean size (shape) of all child maps within the hierarchical neural network.

    Parameters:
    - map (MapNode): The root neuron of the hierarchical neural network.

    Returns:
    - numpy.ndarray: The mean size (shape) of all child maps in the hierarchy.
    """
    mean = np.asarray(get_child_map_sizes(map)).mean(axis=0)
    logger.info(f"Average Child Map Weights: {mean}")
    return mean
