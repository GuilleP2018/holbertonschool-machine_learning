#!/usr/bin/env python3
"""This modlue containes the function forward that performs the forward
algorithm for a hidden markov model"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Thias function calculates the forward algorithm for a hidden markov
    model
    """
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if np.all(np.isclose(np.sum(Emission, axis=1), 1)) is False:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None
    if Initial.shape[1] != 1:
        return None, None
    if np.all(np.isclose(np.sum(Initial), 1)) is False:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if Transition.shape[0] != Initial.shape[0]:
        return None, None
    if Transition.shape[1] != Initial.shape[0]:
        return None, None
    if np.all(np.isclose(np.sum(Transition, axis=1), 1)) is False:
        return None, None

    hidden_states = Initial.shape[0]
    observations = Observation.shape[0]
    F = np.zeros((hidden_states, observations))
    index = Observation[0]
    E = Emission[:, index]
    F[:, 0] = Initial.T * E
    for i in range(1, observations):
        for j in range(hidden_states):
            F[j, i] = np.sum(F[:, i-1] * Transition[
                :, j] * Emission[j, Observation[i]])

    P = np.sum(F[:, observations-1], axis=0)

    return P, F
