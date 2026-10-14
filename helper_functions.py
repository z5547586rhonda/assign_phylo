"""Helper functions for HW3"""
import numpy as np
from copy import deepcopy
from matplotlib.axes import Axes


class Node:
    def __init__(
        self,
        name: str,
        left: "Node",
        left_distance: float,
        right: "Node",
        right_distance: float,
        confidence: float = None,
    ):
        """A node in a binary tree produced by neighbor joining algorithm.

        Parameters
        ----------
        name: str
            Name of the node.
        left: Node
            Left child.
        left_distance: float
            The distance to the left child.
        right: Node
            Right child.
        right_distance: float
            The distance to the right child.
        confidence: float
            The confidence level of the split determined by the bootstrap method.
            Only used if you implement Bonus Problem 1.

        Notes
        -----
        The current public API needs to remain as it is, i.e., don't change the
        names of the properties in the template, as the tests expect this kind
        of structure. However, feel free to add any methods/properties/attributes
        that you might need in your tree construction.

        """
        self.name = name
        self.left = left
        self.left_distance = left_distance
        self.right = right
        self.right_distance = right_distance
        self.confidence = confidence


def neighbor_joining(distances: np.ndarray, labels: list) -> Node:
        matrix = distances.copy().astype(float)
        nodes = []
        for label in labels:
            nodes.append(Node(label, None, 0, None, 0))
    
        while len(nodes) > 2:
            n = len(nodes)
            total_distances = np.sum(matrix, axis=1)
            q_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        q_matrix[i][j] = (n - 2) * matrix[i][j] -  total_distances[i] - total_distances[j]
                    else:
                        q_matrix[i][j] = np.inf
    
            min_index = np.argmin(q_matrix)

            i = min_index // n
            j = min_index % n
    
            distance_ij = matrix[i][j]
            left_distance = distance_ij / 2 + (total_distances[i] - total_distances[j]) / (2 * (n - 2))
            right_distance = distance_ij - left_distance

            new_node = Node("U" + str(n), nodes[i], left_distance, nodes[j], right_distance)
    
            new_distances = []
    
            for k in range(n):
                if k != i and k != j:
                    distance = (matrix[i][k] + matrix[j][k] - matrix[i][j]) / 2
                    new_distances.append(distance)
    
            remaining_nodes = []
            
            for k in range(n):
                if k != i and k != j:
                    remaining_nodes.append(k)
    
            new_n = len(remaining_nodes) + 1
            new_matrix = np.zeros((new_n, new_n))

            for a in range(len(remaining_nodes)):
                for b in range(len(remaining_nodes)):
                    new_matrix[a][b] = matrix[remaining_nodes[a]][remaining_nodes[b]]
    
            for k in range(len(new_distances)):
                new_matrix[k][-1] = new_distances[k]
                new_matrix[-1][k] = new_distances[k]
    
            matrix = new_matrix
            new_nodes = []
    
            for k in remaining_nodes:
                new_nodes.append(nodes[k])
    
            new_nodes.append(new_node)
            nodes = new_nodes
    
        distance = matrix[0][1]
        left_distance = distance / 2
        right_distance = distance / 2
        root = Node("root", nodes[0], left_distance, nodes[1], right_distance)
        return root


def plot_nj_tree(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    nodes = [tree]
    x_positions = [0]
    positions = {}

    i = 0
    while i < len(nodes):
        node = nodes[i]
        x = x_positions[i]

        if node.left is not None:
            nodes.append(node.left)
            x_positions.append(x + node.left_distance)

        if node.right is not None:
            nodes.append(node.right)
            x_positions.append(x + node.right_distance)

        i += 1

    stack = [tree]
    leaf_number = 0
    
    while len(stack) > 0:
        node = stack.pop()
        if node.left is None and node.right is None:
            positions[node] = leaf_number
            leaf_number += 1
        else:
            if node.right is not None:
                stack.append(node.right)

            if node.left is not None:
                stack.append(node.left)

    for i in range(len(nodes) - 1, -1, -1):
        node = nodes[i]
        x = x_positions[i]

        if node.left is not None and node.right is not None:
            left_y = positions[node.left]
            right_y = positions[node.right]

            y = (left_y + right_y) / 2
            positions[node] = y

            left_x = x + node.left_distance
            right_x = x + node.right_distance

            ax.plot([x, left_x], [left_y, left_y], color="black", linewidth=0.8)
            ax.plot([x, right_x], [right_y, right_y], color="black", linewidth=0.8)

            ax.plot([x, x], [left_y, right_y], color="black", linewidth=0.8)

            ax.text((x + left_x) / 2, left_y - 0.1, f"{node.left_distance:.2f}", fontsize=7, ha="center")
            ax.text((x + right_x) / 2, right_y - 0.1, f"{node.right_distance:.2f}", fontsize=7, ha="center")
        else:
            ax.text(x + 0.2, positions[node], node.name, va="center")

    ax.set_xlabel("Genetic distance")
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.margins(x=0.15)

    return

def _find_a_parent_to_node(tree: Node, node: Node) -> tuple:
    """Utility function for reroot_tree"""
    stack = [tree]

    while len(stack) > 0:

        current_node = stack.pop()
        if node.name == current_node.left.name:
            return current_node, "left"
        elif node.name == current_node.right.name:
            return current_node, "right"

        stack += [
            n for n in [current_node.left, current_node.right] if n.left is not None
        ]

    return None



def _remove_child_from_parent(parent_node: Node, child_location: str) -> None:
    """Utility function for reroot_tree"""
    setattr(parent_node, child_location, None)
    setattr(parent_node, f"{child_location}_distance", 0.0)


def reroot_tree(original_tree: Node, outgroup_node: Node) -> Node:
    """A function to create a new root and invert a tree accordingly.

    This function reroots tree with nodes in original format. If you
    added any other relational parameters to your nodes, these parameters
    will not be inverted! You can modify this implementation or create
    additional functions to fix them.

    Parameters
    ----------
    original_tree: Node
        A root node of the original tree.
    outgroup_node: Node
        A Node to set as an outgroup (already included in a tree).
        Find it by it's name and then use it as parameter.

    Returns
    -------
    Node
        Inverted tree with a new root node.
    """
    tree = deepcopy(original_tree)

    parent, child_loc = _find_a_parent_to_node(tree, outgroup_node)
    distance = getattr(parent, f"{child_loc}_distance")
    _remove_child_from_parent(parent, child_loc)

    new_root = Node("new_root", parent, distance / 2, outgroup_node, distance / 2)
    child = parent

    while tree != child:
        parent, child_loc = _find_a_parent_to_node(tree, child)

        distance = getattr(parent, f"{child_loc}_distance")
        _remove_child_from_parent(parent, child_loc)

        empty_side = "left" if child.left is None else "right"
        setattr(child, f"{empty_side}_distance", distance)
        setattr(child, empty_side, parent)

        if tree.name == parent.name:
            break
        child = parent

    other_child_loc = "right" if child_loc == "left" else "left"
    other_child_distance = getattr(parent, f"{other_child_loc}_distance")

    setattr(child, f"{empty_side}_distance", other_child_distance + distance)
    setattr(child, empty_side, getattr(parent, other_child_loc))

    return new_root


def sort_children_by_leaves(tree: Node) -> None:
    """Sort the children of a tree by their corresponding number of leaves.

    The tree can be changed inplace.

    Parameters
    ----------
    tree: Node
        The root node of the tree.

    """
    raise NotImplementedError()


def plot_nj_tree_radial(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram
    with a radial layout.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree_radial(tree=tree, ax=ax)
    >>> fig.savefig("example_radial.png")

    """
    raise NotImplementedError()