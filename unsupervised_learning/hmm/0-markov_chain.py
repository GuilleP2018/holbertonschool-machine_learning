#!/usr/bin/env python3
"""This modlue contains the function markov_chain that
determines the probability"""
import numpy as np


def markov_chain(P, s, t=1):
    """This function determines the probability of a markov chain being in a
    particular state after a specified number of iterations
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if not isinstance(s, np.ndarray) or s.ndim != 2:
        return None
    if s.shape[0] != 1 or s.shape[1] != P.shape[0]:
        return None
    if not isinstance(t, int) or t <= 0:
        return None
    probs = P.shape[0]
    if not np.all(P >= 0):
        return None
    if not np.all(np.isclose(P.sum(axis=1), 1)):
        return None
    for i in range(t):
        s = np.matmul(s, P)
    return s
