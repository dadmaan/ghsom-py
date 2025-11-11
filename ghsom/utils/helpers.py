import random
import re

import numpy as np
import pandas as pd

from ghsom.io.parsing import encode_ghsom_node


# MARK: Hierarchy
def get_used_neurons(root, dataset, labels):
    """
    Retrieves the count of used units for each label in the GHSOM for a given dataset.

    Parameters:
    - root (MapNode): The root neuron of the GHSOM.
    - dataset (list): The input dataset used for training or evaluation.
    - labels (list): The corresponding labels for the input dataset.

    Returns:
    - dict: A nested dictionary where the outer dictionary keys are neurons, and the inner dictionary
            keys are labels. The values represent a tuple (count, list of idx) of occurrences of each label
            in the respective neuron.
    """
    used_neurons = dict()
    for data, label in zip(dataset, labels):
        neuron = root
        while neuron.child_map is not None:
            map = neuron.child_map
            neuron = map.winner_neuron(data)[0][0]

        if neuron not in used_neurons:
            used_neurons[neuron] = dict()
        if label not in used_neurons[neuron]:
            used_neurons[neuron][label] = 0
        used_neurons[neuron][label] += 1

    return used_neurons


def get_random_sample_used_neurons(ghsom, dataset, labels, sample_size=1):
    """
    Randomly selects a subset of the input dataset, gets the used neurons for the subset,
    and randomly samples a specified number of neurons from the used neurons.

    Parameters:
    - ghsom (MapNode): The root neuron of the hierarchical neural network.
    - dataset (list): The input dataset used for training or evaluation.
    - labels (list): The corresponding labels for the input dataset.
    - sample_size (int): The number of neurons to sample from the used neurons. Default is 1.

    Returns:
    - tuple: A tuple (sample_neurons, genre_tags) representing the sampled neurons and their
             corresponding genre tags. sample_neurons is a list of the sampled neurons, and
             genre_tags is a list of dictionaries, where each dictionary is the output of the
             get_used_neurons method for the corresponding sampled neuron.
    """
    # Determine the smallest length between dataset and labels to avoid IndexError
    min_length = min(len(dataset), len(labels))

    # Randomly select a subset of the dataset
    subset_indices = random.sample(range(min_length), int(min_length * 0.5))
    subset_dataset = [dataset[idx] for idx in subset_indices]
    subset_labels = [labels[idx] for idx in subset_indices]

    # Get the used neurons for the subset
    used_neurons = get_used_neurons(ghsom, subset_dataset, subset_labels)

    # Randomly sample neurons from the used neurons
    sample_neurons = random.sample(list(used_neurons.keys()), sample_size)

    # Get the genre tags for the sampled neurons
    genre_tags = [used_neurons[sample] for sample in sample_neurons]

    return sample_neurons, genre_tags


def get_reference_data_point(neuron, df_reference):
    """
    Gets the reference data points corresponding to the input dataset of a given neuron.

    Parameters:
    - neuron (MapNode): The neuron to get the reference data points for.
    - df_reference (pandas.DataFrame): The dataset used to train the GHSOM model.

    Returns:
    - list: A list of reference data points corresponding to the input dataset of the given neuron.
    - list: A list of index values corresponding to the reference data points.
    """
    samples = []
    idx = []
    for input in neuron.input_dataset:
        # Find the row in the reference dataset that matches the input
        row = df_reference[
            (df_reference.dim1 == input[0]) & (df_reference.dim2 == input[1])
        ]
        if not row.empty:
            # Assuming that each input corresponds to a unique row in df_reference
            index_value = row.index.item()  # Get the index as an integer
            idx.append(index_value)
            samples.append(
                row.values[0]
            )  # Assuming we want the first (and only) row of values

    return samples, idx


def find_best_matching_unit(ghsom, data_point):
    """
    Finds the best matching unit (BMU) for a given data point in a GHSOM and updates the BMU's level information.

    This function traverses down the GHSOM hierarchy to find the winning neuron (BMU) for the data point.
    It also updates the BMU's string representation to reflect the correct level in the hierarchy where the BMU was found.

    Parameters:
    - ghsom (MapNode): The root neuron of the GHSOM.
    - data_point (list): The data point to find the BMU for.

    Returns:
    - tuple: A tuple containing the parent BMU, the BMU neuron, and the updated string representation of the BMU.
    """
    _neuron = ghsom
    # Get the parent node
    parent = _neuron.child_map.winner_neuron(data_point)[0][0]

    # Traverse down the GHSOM hierarchy to find the winning neuron for the data point
    # Count number of times traversing the GHSOM hierarchy starting at root (zero unit)
    level = 0
    while _neuron.child_map is not None:
        _gsom = _neuron.child_map
        _neuron = _gsom.winner_neuron(data_point)[0][0]
        level += 1

    # Define the regular expression pattern to match 'level' followed by a space and a number
    pattern = r"(level\s)(\d+)"

    # Replacement string, which includes the new level value
    replacement = r"\g<1>{}".format(level)

    # Substitute the new level value
    exact_location = re.sub(pattern, replacement, str(_neuron))

    # Remove newline characters from the updated string
    exact_location = exact_location.rstrip(" \n")

    return parent, _neuron, exact_location


def assign_ghsom_clusters(data_train, ghsom_root_node, lookup_table):
    """
    Assigns GHSOM cluster IDs to each sample in the data_train DataFrame.

    Parameters:
    - data_train (pandas.DataFrame): The input DataFrame containing the training data.
    - ghsom_root_node (GHSOM): Root node of the trained GHSOM model.
    - lookup_table (dict): The lookup table containing node_ids and corresponding GHSOM nodes.

    Returns:
    - pandas.DataFrame: A copy of the input DataFrame with an additional column 'GHSOM_cluster' containing the cluster IDs.
    """
    # Create a copy of the input DataFrame to avoid modifying the original data
    data = data_train.copy()

    # Ensure the column 'GHSOM_cluster' is of integer type
    if "GHSOM_cluster" in data.columns:
        data["GHSOM_cluster"] = pd.to_numeric(
            data["GHSOM_cluster"], downcast="integer", errors="coerce"
        )
    else:
        # If the column does not exist, initialize it as integer with NaNs which will be filled later
        data["GHSOM_cluster"] = pd.Series(dtype="Int64")

    # Iterate over the DataFrame and find the best matching unit for each sample
    for idx, v in enumerate(data_train.values):
        _, _, nr = find_best_matching_unit(ghsom_root_node, v)
        # Use .at for faster access if you're updating a single value
        data.at[idx, "GHSOM_cluster"] = encode_ghsom_node(lookup_table, nr)

    return data


def create_neuron_table(data_train, ghsom_root_node, lookup_table):
    """
    Creates a table mapping GHSOM cluster IDs to their corresponding neuron representations and neurons.
    This function traverses the GHSOM hierarchy starting from the root node to find the best matching unit
    for each sample in the data_train. It then creates a neuron table with these mappings.

    Parameters:
    - data_train (pandas.DataFrame): A DataFrame containing the GHSOM training data.
    - ghsom_root_node (GHSOMNode): The root node of the GHSOM model. This node is the entry point
      for traversing the GHSOM hierarchy.
    - lookup_table (dict): A dictionary of mappings of GHSOM cluster IDs and neuron representations.

    Returns:
    - dict: A dictionary where each key is a GHSOM cluster ID (node ID) and each value is a tuple containing
      the neuron representation and the neuron object. This table provides a direct lookup from a cluster ID
      to its corresponding neuron details.

    """
    data = data_train.copy()
    neuron_table = {}

    for value in data.values:
        _, neuron, neuron_representation = find_best_matching_unit(ghsom_root_node, value)
        node_id = encode_ghsom_node(lookup_table, neuron_representation)
        neuron_table[node_id] = (neuron_representation, neuron)

    return neuron_table


# MARK: Cluster
def calculate_cosine_similarity_between_two_neurons(neuron_a, neuron_b):
    """
    Calculates the cosine similarity between two vectors.

    Parameters:
    - vector_a (array-like): The weight vector of the first neuron.
    - vector_b (array-like): The weight vector of the second neuron.

    Returns:
    - float: The cosine similarity between the two vectors. This value ranges from -1 (exactly opposite)
             to 1 (exactly the same), with 0 indicating orthogonality (decorrelation).

    Raises:
    - ValueError: If the vectors do not have the same length.
    """

    # Convert vectors to numpy arrays if they aren't already
    weight_vector_a = neuron_a.weight_vector()
    weight_vector_b = neuron_b.weight_vector()

    # Calculate the dot product of vectors
    dot_product = np.dot(weight_vector_a, weight_vector_b)

    # Calculate the norm (magnitude) of each vector
    norm_a = np.linalg.norm(weight_vector_a)
    norm_b = np.linalg.norm(weight_vector_b)

    # Avoid division by zero
    if norm_a == 0 or norm_b == 0:
        raise ValueError(
            "One of the vectors is zero, cannot compute cosine similarity."
        )

    # Compute cosine similarity
    cosine_sim = dot_product / (norm_a * norm_b)

    return cosine_sim


def get_neurons_similarity_matrix(neuron, ghsom_deep, neuron_table, sort=True):
    """
    Calculates distances and neighborhood relationships between a given neuron and all other neurons
    in a GHSOM model.

    Parameters:
    - neuron (int or GHSOMNode): The neuron for which relationships are to be calculated. This can be
                                either an integer index or a GHSOMNode object.
    - ghsom_deep (GHSOM): The GHSOM model containing the neurons.
    - neuron_table (dict): A lookup table mapping neuron indices to tuples containing neuron IDs and GHSOMNode objects.

    Returns:
    - dict: A dictionary where keys are neuron indices and values are tuples containing:
            - The weight distance from the given neuron to the other neuron.
            - A boolean indicating if the two neurons are neighbors.
            - A boolean indicating if the two neurons are in the same column.
            - A boolean indicating if the two neurons are in the same row.
    """
    # Access the child map of the GHSOM model
    _gsom = ghsom_deep.child_map
    distances = {}

    # If the neuron parameter is an integer, retrieve the corresponding GHSOMNode from the neuron_table
    if isinstance(neuron, int):
        neuron = neuron_table[neuron][1]

    # Iterate over all neurons in the neuron_table to calculate distances and relationships
    for idx, n in neuron_table.items():
        distances[idx] = (
            neuron.weight_distance_from_other_unit(n[1]),
            calculate_cosine_similarity_between_two_neurons(neuron, n[1]),
            _gsom.are_neurons_neighbours(neuron, n[1]),
            _gsom.are_in_same_column(neuron, n[1]),
            _gsom.are_in_same_row(neuron, n[1]),
        )

    if sort:
        distances = sorted(distances.items(), key=lambda item: item[1][0])
        distances = {node[0]: node[1] for node in distances}
    return distances


def get_path_to_node(distances, target_node_id):
    """
    Identifies a specific node_id in the sorted list of distances and returns all items from the top
    to that specific node_id.

    Parameters:
    - sorted_distances (list): A sorted list of tuples, where each tuple contains:
                               (neuron index, (distance, neighbors, same column, same row))
    - target_node_id (str): The neuron index of the target node.

    Returns:
    - list: A list of tuples from the top of the sorted list to the specified target_node_id, inclusive.
            Returns an empty list if the target_node_id is not found.
    """
    path_to_node = []
    for idx, details in distances.items():
        path_to_node.append((idx, details))
        if idx == target_node_id:
            return path_to_node
    return []


# MARK: Traversing Latent Space
# def greedy_traverse_ghsom(node_distances, start_id, end_id):
#     """
#     Traverses the GHSOM hierarchy from the start node to the end node using a greedy algorithm.

#     Parameters:
#     - node_distances (list of tuples): Each tuple contains the node ID and a tuple of
#       (distance, is_neighbour, is_same_column, is_same_row).
#     - start_id (int): The ID of the start node.
#     - end_id (int): The ID of the end node.

#     Returns:
#     - list: The path from the start node to the end node as a list of node IDs.
#     """
#     # Convert the list to a dictionary for easier access
#     if isinstance(node_distances, list):
#         node_dict = {node[0]: node[1] for node in node_distances}

#     node_dict = node_distances
#     # Initialize the path with the start node
#     path = [start_id]
#     current_id = start_id

#     # Continue until the end node is reached
#     while current_id != end_id:
#         # Get the current node details
#         current_node = node_dict[current_id]

#         # Find the closest neighbor that moves towards the end node and hasn't been visited
#         neighbors = [(id, details) for id, details in node_dict.items() if id not in path]

#         # Sort neighbors by distance
#         neighbors.sort(key=lambda x: x[1][0])

#         # Check if there's a valid neighbor to move to
#         if not neighbors:
#             raise ValueError("No path found to the end node.")

#         # Move to the closest valid neighbor
#         current_id = neighbors[0][0]
#         path.append(current_id)

#     return path

# import heapq

# def dijkstra_path(node_distances, start_id, end_id):
#     """
#     Finds the shortest path from the start node to the end node using Dijkstra's algorithm.

#     Parameters:
#     - node_distances (list of tuples): Each tuple contains the node ID and a tuple of
#       (distance, is_neighbour, is_same_column, is_same_row).
#     - start_id (int): The ID of the start node.
#     - end_id (int): The ID of the end node.

#     Returns:
#     - list: The shortest path from the start node to the end node as a list of node IDs.
#     """
#     # If list, convert it to a dictionary for easier access
#     if isinstance(node_distances, list):
#         node_dict = {node[0]: node[1] for node in node_distances}

#     graph = {node[0]: {} for node in node_dict.items()}

#     for idx, node in node_dict.items():
#         node_id = idx
#         for _idx, other_node in node_dict.items():
#             other_node_id = _idx
#             if node_id != other_node_id:
#                 # Use the distance as the weight of the edge
#                 graph[node_id][other_node_id] = other_node[0]


#     # Priority queue to select the node with the smallest distance
#     priority_queue = [(0, start_id)]
#     # Distances to nodes initialized to infinity except the start node
#     distances = {node: float('inf') for node in graph}
#     distances[start_id] = 0
#     # Dictionary to store the path
#     previous_nodes = {node: None for node in graph}


#     while priority_queue:

#         current_distance, current_node = heapq.heappop(priority_queue)

#         # If we reach the end node, reconstruct the path
#         if current_node == end_id:
#             print(previous_nodes)
#             path = []
#             while previous_nodes[current_node] is not None:
#                 path.insert(0, current_node)
#                 current_node = previous_nodes[current_node]
#             path.insert(0, start_id)
#             return path

#         # Nodes can get added to the priority queue multiple times. We only
#         # process a vertex the first time we remove it from the priority queue.
#         if current_distance > distances[current_node]:
#             continue

#         for neighbor, weight in graph[current_node].items():
#             distance = current_distance + weight

#             # Only consider this new path if it's better
#             if distance < distances[neighbor]:
#                 distances[neighbor] = distance
#                 previous_nodes[neighbor] = current_node
#                 heapq.heappush(priority_queue, (distance, neighbor))

#     return []


# =================================================
# ================= Extra stuff ===================
# =================================================
def collect_neurons(node):
    def _collect_neurons(node, collected_neurons):
        """
        Recursively collects all neurons from a GHSOM hierarchy starting from the given node.

        Parameters:
        - node: The current node in the GHSOM hierarchy.
        - collected_neurons (list): The list where neurons are collected.

        Example:
        collected_neurons = []
        collect_neurons(ghsom_deep, collected_neurons)

        """
        if node.child_map:
            for _, child_neuron in node.child_map.neurons.items():
                _collect_neurons(child_neuron, collected_neurons)
        else:
            collected_neurons.append(node)

    collected_neurons = []
    _collect_neurons(node, collected_neurons)

    return collected_neurons


def _get_used_neurons_postion(neuron, data, label, level, used_units, parent="root"):
    """
    Recursively traverses a GHSOM to count the number of times each neuron is used for a given dataset.

    Parameters:
    - neuron (MapNode): The neuron to start the traversal from.
    - data (list): The input data point to count the neuron usage for.
    - label: The label corresponding to the input data point.
    - level (int): The level of the neuron in the GHSOM hierarchy.
    - used_units (dict): A dictionary to store the neuron usage counts.

    Returns:
    - None
    """
    neuron_pos = neuron.position
    if f"{parent}_{level}_{neuron_pos}" not in used_units:
        used_units[f"{parent}_{level}_{neuron_pos}"] = dict()

    if label not in used_units[f"{parent}_{level}_{neuron_pos}"]:
        used_units[f"{parent}_{level}_{neuron_pos}"][label] = 0

    used_units[f"{parent}_{level}_{neuron_pos}"][label] += 1

    if neuron.child_map is not None:
        # Recursively check child_maps
        child_neuron = neuron.child_map.winner_neuron(data)[0][0]
        _get_used_neurons_postion(
            child_neuron,
            data,
            label,
            level + 1,
            used_units,
            parent=f"{level}_{neuron_pos}",
        )


def get_used_neurons_postion(root, dataset, labels):
    """
    Counts the number of times each neuron in a GHSOM is used for a given dataset.

    Parameters:
    - root (MapNode): The root neuron of the GHSOM.
    - dataset (list): The input dataset to count the neuron usage for.
    - labels (list): The corresponding labels for the input dataset.

    Returns:
    - dict: A dictionary where the keys are the positions of the neurons, and the values are
            dictionaries where the keys are the labels and the values are the number of times
            the corresponding neuron was used for the corresponding label.
    """
    used_units = dict()
    for data, label in zip(dataset, labels):
        neuron = root
        _get_used_neurons_postion(neuron, data, label, 0, used_units)

    return used_units
