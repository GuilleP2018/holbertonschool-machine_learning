#!/usr/bin/env python3
"""This modlue contains the function regular that determines
the steady state"""
import numpy as np


def regular(P):
    """This function determines the steady state probabilities of a regular
    markov chain
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None

    probs = P.shape[0]
    if not np.all(P >= 0):
        return None
    if not np.all(np.isclose(P.sum(axis=1), 1)):
        return None

    starting_probs = np.full(probs, 1 / probs)[np.newaxis, ...]
    P_copy = np.copy(P)
    steady_state = np.copy(starting_probs)

    while True:
        P_copy = np.matmul(P_copy, P)
        if np.any(P_copy <= 0):
            return None
        starting_probs = np.matmul(starting_probs, P)

        if np.all(starting_probs == steady_state):
            return starting_probs

        steady_state = np.copy(starting_probs)
