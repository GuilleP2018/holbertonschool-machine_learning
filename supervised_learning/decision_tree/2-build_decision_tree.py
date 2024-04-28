#!/usr/bin/env python3
"""Build a decision tree and add max_depth_below function to it"""
import numpy as np


class Node:
    """Node class for a decision tree"""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize a node in a decision tree"""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Calculate the maximum depth below this node"""
        if self.is_leaf:
            return self.depth
        else:
            return max(self.left_child.max_depth_below(),
                       self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """Count the number of nodes below this node"""
        if only_leaves:
            count = self.left_child.count_nodes_below(only_leaves=True)
            count += self.right_child.count_nodes_below(only_leaves=True)
            return count
        else:
            count = 1 + self.left_child.count_nodes_below()
            count += self.right_child.count_nodes_below()
            return count

    def __str__(self):
        """__str__ method for Node class"""
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

    def left_child_add_prefix(self, text):
        """Add prefix to the text of left child"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        new_text += "\n".join(["    |  " + line for line in lines[1:-1]])
        new_text += "\n" if len(lines) > 1 else ""
        return new_text

    def right_child_add_prefix(self, text):
        """Add prefix to the text of right child"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        new_text += "\n".join(["     " + "  " + line for line in lines[1:-1]])
        new_text += "\n" if len(lines) > 1 else ""
        return new_text


class Leaf(Node):
    """Leaf class for a decision tree"""
    def __init__(self, value, depth=None):
        """Initialize a leaf node in a decision tree"""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """max_depth_below for a leaf node is its own depth"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Count the number of nodes below this node"""
        return 1

    def __str__(self):
        """__str__ method for Leaf class"""
        return (f"-> leaf [value={self.value}] ")


class Decision_Tree():
    """Decision Tree class"""
    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
        """Initialize a decision tree"""
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
        """Calculate the depth of the decision tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count the number of nodes in the decision tree"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """__str__ method for Decision_Tree class"""
        return self.root.__str__()
