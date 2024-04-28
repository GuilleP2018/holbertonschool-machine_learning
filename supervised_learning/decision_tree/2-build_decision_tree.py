#!/usr/bin/env python3
"""Module for the Decision_Tree class."""
import numpy as np


class Node:
    """A class representing a node in a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize a Node.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Calculate the maximum depth of the tree below this node.
        """
        if self.is_leaf:
            return self.depth
        else:
            if self.left_child:
                left_depth = self.left_child.max_depth_below()
            else:
                left_depth = 0
            if self.right_child:
                right_depth = self.right_child.max_depth_below()
            else:
                right_depth = 0
            return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Count the number of nodes below this node,
        optionally excluding non-leaf nodes.
        """
        if only_leaves:
            count = self.left_child.count_nodes_below(only_leaves=True)
            count += self.right_child.count_nodes_below(only_leaves=True)
            return count
        else:
            count = 1 + self.left_child.count_nodes_below()
            count += self.right_child.count_nodes_below()
            return count

    def left_child_add_prefix(self, text):
        """ Add prefix to the left child """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        new_text += "\n".join(["    |  " + line for line in lines[1:-1]])
        new_text += "\n" if len(lines) > 1 else ""
        return new_text

    def right_child_add_prefix(self, text):
        """ Add prefix to the right child """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        new_text += "\n".join(["     " + "  " + line for line in lines[1:-1]])
        new_text += "\n" if len(lines) > 1 else ""
        return new_text

    def __str__(self):
        """
        Method that returns the string representation of the current node
        """
        node_str = (
            f"root [feature={self.feature}, threshold={self.threshold}]\n"
            if self.is_root else
            f"-> node [feature={self.feature}, "
            f"threshold={self.threshold}]\n"
        )

        if self.is_leaf:
            return node_str

        left_str = self.left_child_add_prefix(
            self.left_child.__str__()) if self.left_child else ""
        right_str = self.right_child_add_prefix(
            self.right_child.__str__()) if self.right_child else ""

        return node_str + left_str + right_str


class Leaf(Node):
    """A class representing a leaf node in a decision tree."""

    def __init__(self, value, depth=None):
        """Initialize a Leaf.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Calculate the maximum depth of the tree below this node.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """This method counts the number of nodes below this node."""
        return 1

    def __str__(self):
        """Return a string representation of the node."""
        return (f"-> leaf [value={self.value}] ")


class Decision_Tree():
    """A class representing a decision tree."""

    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
        """Initialize a Decision_Tree.
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Calculate the maximum depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count the number of nodes in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Return a string representation of the tree."""
        return self.root.__str__()
