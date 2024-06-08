def viterbi(states, observations, start_prob, trans_prob, emission_prob):
    # Initialize Viterbi matrix and backpointer matrix
    V = [{}]
    backpointer = [{}]

    # Initialize base cases (t == 0)
    for state in states:
        V[0][state] = start_prob[state] * emission_prob[state][observations[0]]
        backpointer[0][state] = None

    # Run Viterbi for t > 0
    for t in range(1, len(observations)):
        V.append({})
        backpointer.append({})

        for state in states:
            max_tr_prob = max(V[t-1][prev_state] * trans_prob[prev_state][state] for prev_state in states)
            for prev_state in states:
                if V[t-1][prev_state] * trans_prob[prev_state][state] == max_tr_prob:
                    max_prob = max_tr_prob * emission_prob[state][observations[t]]
                    V[t][state] = max_prob
                    backpointer[t][state] = prev_state
                    break

    # Find the final most probable state
    max_prob = max(value for value in V[-1].values())
    previous = None

    for state, data in V[-1].items():
        if data == max_prob:
            best_state = state
            break

    # Path backtracking
    opt_path = [best_state]
    for t in range(len(V) - 1, 0, -1):
        opt_path.insert(0, backpointer[t][opt_path[0]])

    return opt_path, max_prob

# Example usage
states = ('Healthy', 'Fever')
observations = ('normal', 'cold', 'dizzy')
start_prob = {'Healthy': 0.6, 'Fever': 0.4}
trans_prob = {
    'Healthy': {'Healthy': 0.7, 'Fever': 0.3},
    'Fever': {'Healthy': 0.4, 'Fever': 0.6},
}
emission_prob = {
    'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
    'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
}

opt_path, max_prob = viterbi(states, observations, start_prob, trans_prob, emission_prob)
print("Most probable path:", opt_path)
print("Probability of the path:", max_prob)
