#!/usr/bin/env python3
"""This modlue contains the function absorbing that determines if a
markov chain is absorbing"""
import numpy as np


def absorbing(P):
    """This function determines if a markov chain is absorbing
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None

    num_states = P.shape[0]

    if not np.isclose(np.sum(P, axis=1), np.ones(num_states))[0]:
        return None
    if np.all(np.diag(P) != 1):
        return False
    if np.all(np.diag(P) == 1):
        return True

    for i in range(num_states):
        if np.any(P[i, :] == 1):
            continue
        break

    sub_matrix_I = P[:i, :i]
    identity_matrix = np.identity(num_states - i)
    sub_matrix_R = P[i:, :i]
    sub_matrix_Q = P[i:, i:]

    try:
        fundamental_matrix = np.linalg.inv(identity_matrix - sub_matrix_Q)
    except Exception:
        return False

    FR_product = np.matmul(fundamental_matrix, sub_matrix_R)
    limiting_matrix = np.zeros((num_states, num_states))
    limiting_matrix[:i, :i] = sub_matrix_I
    limiting_matrix[i:, :i] = FR_product
    sub_matrix_Qbar = limiting_matrix[i:, i:]

    if np.all(sub_matrix_Qbar == 0):
        return True

    return False
