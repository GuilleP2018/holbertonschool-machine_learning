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
