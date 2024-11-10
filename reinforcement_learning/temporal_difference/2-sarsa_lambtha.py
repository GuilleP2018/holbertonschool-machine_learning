#!/usr/bin/env python3
"""This module contains the function for the SARSA(λ) algorithm."""
import numpy as np


def sarsa_lambtha(
    env,
    Q,
    lambtha,
    episodes=5000,
    max_steps=100,
    alpha=0.1,
    gamma=0.99,
    epsilon=1,
    min_epsilon=0.1,
    epsilon_decay=0.05,
):
    """
    Performs the SARSA(λ) algorithm to update the Q table.
    """

    def epsilon_greedy(state, Q, epsilon):
        """Selects an action using epsilon-greedy policy."""
        if np.random.rand() < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(Q[state])

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        action = epsilon_greedy(state, Q, epsilon)

        E = np.zeros_like(Q)

        for step in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]

            next_action = epsilon_greedy(next_state, Q, epsilon)

            delta = reward + gamma * Q[next_state,
                                       next_action] - Q[state, action]

            E[state, action] += 1

            Q += alpha * delta * E
            E *= gamma * lambtha

            if terminated or truncated:
                break

            state = next_state
            action = next_action

        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay * episode))

    return Q
