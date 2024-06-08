import numpy as np

def viterbi(obs, states, start_p, trans_p, emit_p):
    """
    Implements the Viterbi algorithm to find the most likely sequence of hidden states.

    Args:
        obs: The sequence of observations.
        states: The set of possible hidden states.
        start_p: Initial probabilities of starting in each state.
        trans_p: Transition probability matrix.
        emit_p: Emission probability matrix.

    Returns:
        The most likely sequence of hidden states.
    """

    V = [{}]
    path = {}

    # Initialize base cases (t == 0)
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]

    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}

        for y in states:
            (prob, state) = max(
                (V[t - 1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states
            )
            V[t][y] = prob
            newpath[y] = path[state] + [y]

        # Don't need to remember the old paths
        path = newpath
    
    # Find the most likely ending state and its path
    (prob, state) = max((V[len(obs) - 1][y], y) for y in states)
    return (prob, path[state])

# Example Usage
obs = ('normal', 'cold', 'dizzy')
states = ('Healthy', 'Fever')
start_p = {'Healthy': 0.6, 'Fever': 0.4}
trans_p = {
    'Healthy': {'Healthy': 0.7, 'Fever': 0.3},
    'Fever': {'Healthy': 0.4, 'Fever': 0.6},
}
emit_p = {
    'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
    'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
}

prob, path = viterbi(obs, states, start_p, trans_p, emit_p)
print('The most likely path is:', path, 'with probability:', prob)
