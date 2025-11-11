import logging
import re

logger = logging.getLogger(__name__)


class GHSOMNode:
    """Node in GHSOM hierarchy tree."""

    def __init__(self, position, map_dimensions, input_dataset_size, level):
        self.position = position
        self.map_dimensions = map_dimensions
        self.input_dataset_size = input_dataset_size
        self.level = level
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self):
        return (
            f"position {self.position} -- map dimensions {self.map_dimensions} -- "
            f"input dataset {self.input_dataset_size} element(s) -- level {self.level}"
        )


def parse_ghsom_hierarchy(ghsom_hierarchy):
    root = None
    node_stack = []

    for line in ghsom_hierarchy.strip().split("\n"):
        parts = line.strip().split("--")

        pos_pattern = r"\((-?\d+),\s*(-?\d+)\)"
        position_str = re.findall(pos_pattern, parts[0])
        position_int = [(int(x), int(y)) for x, y in position_str]

        dim_pattern = r"\((-?\d+),\s*(-?\d+),\s*(-?\d+)\)"
        dimension_str = re.findall(dim_pattern, parts[1])
        dimension_int = [(int(x), int(y), int(z)) for x, y, z in dimension_str]

        input_dataset_size = int(parts[2].split()[2])

        level = int(parts[3].split()[1])

        # Create a new node
        new_node = GHSOMNode(
            position_int[0], dimension_int[0], input_dataset_size, level
        )

        # Check if this is the root node
        if level == 0:
            root = new_node
            node_stack = [new_node]
        else:
            # Pop nodes from the stack until we find the parent level
            while node_stack and node_stack[-1].level >= level:
                node_stack.pop()
            # The current top of the stack is the parent
            parent_node = node_stack[-1]
            parent_node.add_child(new_node)
            node_stack.append(new_node)

    return root


def create_lookup_table(root_node, short_id=False):
    """
    Creates a lookup table for the GHSOM hierarchy by assigning a unique ID to each node.

    Parameters:
    - root_node (GHSOMNode): The root node of the GHSOM tree.
    - short_id (bool): If True, creates simple integer ID for each cluster.
                Otherwise, encodes position, map dimensions and level.

    Returns:
    - dict: A lookup table where keys are unique IDs and values are the corresponding nodes.
    """
    lookup_table = {}
    next_id = 1  # Initialize a counter for the next available ID

    def traverse_and_add_to_table(node, lookup_table, next_id, short_id):
        """
        Recursively traverses the GHSOM tree and adds each node to the lookup table with a simple identifier.

        Parameters:
        - node (GHSOMNode): The current node being visited.
        - lookup_table (dict): The lookup table being populated with nodes.
        """

        if node is root_node:
            node_id = "root"
            # Add the current node to the lookup table
            lookup_table[node_id] = node
        else:
            if short_id:
                node_id = next_id
            else:
                # Create identifier using the available ID and the node level, position, dimensions
                node_id = int(
                    f"{next_id}{node.level}{node.position[0]}{node.position[1]}{node.map_dimensions[0]}{node.map_dimensions[1]}{node.map_dimensions[2]}"
                )
            # Add the current node to the lookup table
            lookup_table[node_id] = node

            next_id += 1

        # Recursively add child nodes to the lookup table
        for child in node.children:
            next_id = traverse_and_add_to_table(child, lookup_table, next_id, short_id)

        return next_id

    # Start the tree traversal from the root node
    traverse_and_add_to_table(root_node, lookup_table, next_id, short_id)

    return lookup_table


def encode_ghsom_node(lookup_table, input_node):
    """
    Encodes the given GHSOM node's representation by finding its corresponding key in the lookup table.

    Parameters:
    - lookup_table (dict): The lookup table containing keys as node identifiers and values as node attributes.
    - input_node_str (str): The representation of the GHSOM node to encode.

    Returns:
    - The key corresponding to the input node's position in the lookup table, or None if not found.
    """
    if not isinstance(input_node, str):
        input_node = str(input_node)

    for key, node in lookup_table.items():
        if str(node) == input_node:
            # Found the node, return its key from the lookup table
            return key
    # If the node is not found, return None
    return None


def decode_ghsom_node(lookup_table, node_id):
    """
    Retrieves the GHSOM node object corresponding to a given node ID from the lookup table.

    Parameters:
    - lookup_table (dict): A dictionary where keys are node IDs (as strings) and values are
                           the corresponding GHSOM node objects.
    - node_id (str or int): The unique identifier of the node to be retrieved. If the node ID
                            is provided as an integer, it will be converted to a string before
                            lookup.

    Returns:
    - GHSOMNode or None: The GHSOM node object associated with the given node ID if found in
                         the lookup table, otherwise None.
    """
    return lookup_table.get(node_id)


def get_ghsom_node_statistics(lookup_table, node_id):
    """
    Reports statistics for a specific GHSOM node using its unique ID and the lookup table.

    Parameters:
    - lookup_table (dict): The lookup table containing node IDs and corresponding nodes.
    - node_id (str): The unique ID of the target node.

    Returns:
    - dict: A dictionary containing statistics for the target node, or None if not found.
    """

    # Look up the node in the lookup table using the integer ID
    node = lookup_table.get(node_id)

    # If the node is not found, return None
    if not node:
        return None

    # Gather statistics for the node
    statistics = {
        "node_id": node_id,
        "position": node.position,
        "map_dimensions": node.map_dimensions,
        "input_dataset_size": node.input_dataset_size,
        "level": node.level,
        "number_of_children": len(node.children),
    }

    # If the node has children, calculate additional statistics
    if node.children:
        statistics["children"] = [
            {
                "node_id": encode_ghsom_node(lookup_table, child),
                "position": child.position,
            }
            for child in node.children
        ]
        statistics["total_input_dataset_size_of_children"] = sum(
            child.input_dataset_size for child in node.children
        )

    return statistics


def get_node_relative_path_by_id(lookup_table, node, print_output=True):
    """
    Retrieves the path in the GHSOM hierarchy for a specific node using its unique ID.

    Parameters:
    - root_node (GHSOMNode): The root node of the GHSOM tree.
    - lookup_table (dict): The lookup table containing node IDs and corresponding nodes.
    - node_id (str): The unique ID of the target node or the node representation.

    Returns:
    - str: A string representation of the path from the root node to the target node, or None if not found.
    """
    # Check if ID of the node is given
    if isinstance(node, int):
        # Look up the node in the lookup table using the integer ID
        target_node = lookup_table.get(node)
    else:
        # If not, retrieve the correct node_id
        node_id = encode_ghsom_node(lookup_table, node)
        target_node = lookup_table.get(node_id)

    # If the node is not found, return None
    if not target_node:
        return None

    # Construct the path from the target node to the root node
    path = []
    current_node = target_node
    while current_node is not None:
        if print_output:
            # Calculate the indentation based on the level (4 spaces per level as an example)
            indentation = "    " * current_node.level
            node_representation = (
                f"{indentation}position {current_node.position} -- map dimensions {current_node.map_dimensions} -- "
                f"input dataset {current_node.input_dataset_size} element(s) -- level {current_node.level}"
            )
        else:
            node_representation = (
                f"position {current_node.position} -- map dimensions {current_node.map_dimensions} -- "
                f"input dataset {current_node.input_dataset_size} element(s) -- level {current_node.level}"
            )
        path.insert(
            0, node_representation
        )  # Insert at the beginning to build the path upwards
        # Find the parent of the current node by looking for a node that has the current node as a child
        parent_node = None
        for lookup_node in lookup_table.values():
            if current_node in lookup_node.children:
                parent_node = lookup_node
                break
        current_node = parent_node

    if print_output:
        # Join the path representations into a single string
        path_str = "\n".join(path)
        logger.info(path_str)
    return path


def create_clusters_dict(root_node):
    """
    Creates a dictionary of clusters from the GHSOM tree.

    Parameters:
    - root_node (GHSOMNode): The root node of the GHSOM tree.

    Returns:
    - dict: A dictionary where keys are cluster identifiers and values are cluster attributes.
    """
    clusters_dict = {}

    def traverse_tree(node, clusters_dict, parent_id=None):
        """
        Recursively traverses the GHSOM tree and populates the clusters dictionary.

        Parameters:
        - node (GHSOMNode): The current node in the GHSOM tree.
        - clusters_dict (dict): The dictionary being populated with cluster information.
        - parent_id (str): The identifier of the parent cluster.
        """
        # Create a unique identifier for the cluster, e.g., based on its position and level
        cluster_id = f"level_{node.level}_pos_{node.position}"

        # Add the current cluster to the dictionary
        clusters_dict[cluster_id] = {
            "position": node.position,
            "map_dimensions": node.map_dimensions,
            "input_dataset_size": node.input_dataset_size,
            "level": node.level,
            "parent_id": parent_id,
            "children_ids": [],
        }

        # Recursively add child clusters to the dictionary
        for child in node.children:
            child_id = f"level_{child.level}_pos_{child.position}"
            clusters_dict[cluster_id]["children_ids"].append(child_id)
            traverse_tree(child, clusters_dict, parent_id=cluster_id)

    # Start the tree traversal from the root node
    traverse_tree(root_node, clusters_dict)

    return clusters_dict


def get_clusters_by_level(root_node, target_level):
    """
    Retrieves all clusters from the GHSOM tree that are at a specific level.

    Parameters:
    - root_node (GHSOMNode): The root node of the GHSOM tree.
    - target_level (int): The level of the clusters to retrieve.

    Returns:
    - list: A list of GHSOMNode objects that are at the specified level.
    """
    clusters_at_level = []

    def traverse_and_collect(node, target_level, clusters_at_level):
        """
        Recursively traverses the GHSOM tree and collects nodes at the target level.

        Parameters:
        - node (GHSOMNode): The current node being visited.
        - target_level (int): The level of clusters to collect.
        - clusters_at_level (list): The list being populated with nodes at the target level.
        """
        # If the current node is at the target level, add it to the list
        if node.level == target_level:
            clusters_at_level.append(node)
        # Recursively visit child nodes
        for child in node.children:
            traverse_and_collect(child, target_level, clusters_at_level)

    # Start the tree traversal from the root node
    traverse_and_collect(root_node, target_level, clusters_at_level)

    return clusters_at_level


def get_ghsom_statistics(root_node):
    """
    Reports statistics from the GHSOM tree, such as the number of children for each node,
    the number of clusters at each level, and the total number of nodes.

    Parameters:
    - root_node (GHSOMNode): The root node of the GHSOM tree.

    Returns:
    - dict: A dictionary containing the statistics of the GHSOM tree.
    """
    statistics = {
        "total_nodes": 0,
        "levels": {},
        "max_children": 0,
        "max_input_dataset_size": 0,
    }

    def traverse_tree(node, statistics):
        """
        Recursively traverses the GHSOM tree and collects statistics.

        Parameters:
        - node (GHSOMNode): The current node being visited.
        - statistics (dict): The dictionary being populated with statistics.
        """
        # Increment the total number of nodes
        statistics["total_nodes"] += 1

        # Update the number of clusters at the current level
        level = node.level
        if level not in statistics["levels"]:
            statistics["levels"][level] = 0
        statistics["levels"][level] += 1

        # Update the maximum number of children for any node
        num_children = len(node.children)
        statistics["max_children"] = max(statistics["max_children"], num_children)

        # Update the maximum input dataset size for any node
        statistics["max_input_dataset_size"] = max(
            statistics["max_input_dataset_size"], node.input_dataset_size
        )

        # Recursively visit child nodes
        for child in node.children:
            traverse_tree(child, statistics)

    # Start the tree traversal from the root node
    traverse_tree(root_node, statistics)

    return statistics
