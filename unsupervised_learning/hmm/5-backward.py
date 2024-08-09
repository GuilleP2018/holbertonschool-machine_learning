#!/usr/bin/env python3
"""This modlue containes the function backward that performs the backward
algorithm for a hidden markov model"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """This function performs the backward algorithm for a hidden markov model
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
    B = np.zeros((hidden_states, observations))
    B[:, observations - 1] = 1
    for t in range(observations - 2, -1, -1):
        for s in range(hidden_states):
            B[s, t] = np.sum(B[:, t + 1] * Transition[s, :] *
                             Emission[:, Observation[t + 1]])

    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])

    return P, B
