#!/usr/bin/env python3
"""This mdolue contains the function pdf(X, m, S)"""
import numpy as np


def pdf(X, m, S):
    """This function calculates the probability density
    function of a Gaussian
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None

    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None

    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None

    if S.shape[0] != S.shape[1]:
        return None

    n, d = X.shape

    normalization_factor = 1.0 / np.sqrt(((2 * np.pi) ** d) * np.linalg.det(S))

    mahalanobis_distance = np.matmul(np.linalg.inv(S), (X - m).T)

    exponent_term = np.exp(
        -0.5 * np.sum((X - m).T * mahalanobis_distance, axis=0))

    pdf = normalization_factor * exponent_term

    pdf = np.maximum(pdf, 1e-300)

    return pdf
