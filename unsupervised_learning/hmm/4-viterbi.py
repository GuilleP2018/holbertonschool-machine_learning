#!/usr/bin/env python3
"""This modlue containes the function viterbi that calculates the most likely
sequence of hidden states for a hidden markov model"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """This fucntion calculates the most likely sequence of hidden states for a
    hidden markov model
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
    viterbi = np.zeros((hidden_states, observations))
    viterbi[:, 0] = Initial.T * Emission[:, Observation[0]]
    backpointer = np.zeros((hidden_states, observations))
    for t in range(1, observations):
        for s in range(hidden_states):
            viterbi[s, t] = np.max(viterbi[:, t - 1] * Transition[:, s] *
                                   Emission[s, Observation[t]])
            backpointer[s, t] = np.argmax(
                viterbi[:, t - 1] * Transition[:, s] *
                Emission[s, Observation[t]])
    P = np.max(viterbi[:, observations - 1])
    Last_state = np.argmax(viterbi[:, observations - 1])
    path = [Last_state]
    for t in range(observations - 1, 0, -1):
        path.insert(0, int(backpointer[Last_state, t]))
        Last_state = int(backpointer[Last_state, t])

    return path, P
