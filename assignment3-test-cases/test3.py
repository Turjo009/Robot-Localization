import numpy as np

def parse_input(input_str):
    lines = input_str.strip().split('\n')
    rows, cols = map(int, lines[0].split())
    map_data = []
    for i in range(1, rows + 1):
        map_data.append(list(lines[i].strip().replace(' ', '')))
    num_observations = int(lines[rows + 1])
    observations = []
    for i in range(rows + 2, rows + 2 + num_observations):
        observations.append(lines[i].strip())
    sensor_error_rate = float(lines[rows + 2 + num_observations])
    return rows, cols, map_data, num_observations, observations, sensor_error_rate

def get_neighbors(row, col, rows, cols, map_data):
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r, c = row + dr, col + dc
        if 0 <= r < rows and 0 <= c < cols and map_data[r][c] == '0':
            neighbors.append((r, c))
    return neighbors

def compute_transition_probabilities(rows, cols, map_data):
    trans_prob = {}
    for r in range(rows):
        for c in range(cols):
            if map_data[r][c] == '0':
                neighbors = get_neighbors(r, c, rows, cols, map_data)
                if neighbors:
                    prob = 1 / len(neighbors)
                    trans_prob[(r, c)] = {neighbor: prob for neighbor in neighbors}
    return trans_prob

def compute_emission_probabilities(rows, cols, sensor_error_rate, map_data):
    correct_prob = 1 - sensor_error_rate
    error_prob = sensor_error_rate / 4
    emission_prob = {}
    for r in range(rows):
        for c in range(cols):
            if map_data[r][c] == '0':
                emission_prob[(r, c)] = {
                    '1011': correct_prob if (r > 0 and map_data[r-1][c] == 'X') and (r < rows - 1 and map_data[r+1][c] == '0') and (c > 0 and map_data[r][c-1] == 'X') and (c < cols - 1 and map_data[r][c+1] == '0') else error_prob,
                    '1010': correct_prob if (r > 0 and map_data[r-1][c] == 'X') and (r < rows - 1 and map_data[r+1][c] == '0') and (c > 0 and map_data[r][c-1] == 'X') else error_prob,
                    '1000': correct_prob if (r > 0 and map_data[r-1][c] == 'X') else error_prob,
                    '1100': correct_prob if (r > 0 and map_data[r-1][c] == 'X') and (r < rows - 1 and map_data[r+1][c] == 'X') else error_prob,
                    '0110': correct_prob if (r < rows - 1 and map_data[r+1][c] == 'X') and (c > 0 and map_data[r][c-1] == '0') and (c < cols - 1 and map_data[r][c+1] == 'X') else error_prob,
                    '0000': correct_prob if (
                        (r == 0 or map_data[r-1][c] != 'X') and
                        (r == rows - 1 or map_data[r+1][c] != 'X') and
                        (c == 0 or map_data[r][c-1] != 'X') and
                        (c == cols - 1 or map_data[r][c+1] != 'X')
                    ) else error_prob
                }
    return emission_prob

def viterbi_algorithm(rows, cols, map_data, observations, start_prob, trans_prob, emission_prob):
    num_observations = len(observations)
    V = np.zeros((num_observations, rows, cols))
    backpointer = np.zeros((num_observations, rows, cols, 2), dtype=int)
    
    for r in range(rows):
        for c in range(cols):
            if map_data[r][c] == '0':
                V[0, r, c] = start_prob[(r, c)] * emission_prob[(r, c)].get(observations[0], 0)

    for t in range(1, num_observations):
        for r in range(rows):
            for c in range(cols):
                if map_data[r][c] == '0':
                    max_prob, max_state = 0, (0, 0)
                    for (pr, pc), prob in trans_prob[(r, c)].items():
                        curr_prob = V[t-1, pr, pc] * prob * emission_prob[(r, c)].get(observations[t], 0)
                        if curr_prob > max_prob:
                            max_prob, max_state = curr_prob, (pr, pc)
                    V[t, r, c] = max_prob
                    backpointer[t, r, c] = max_state

    max_prob = 0
    last_state = (0, 0)
    for r in range(rows):
        for c in range(cols):
            if V[-1, r, c] > max_prob:
                max_prob = V[-1, r, c]
                last_state = (r, c)

    best_path = [last_state]
    for t in range(num_observations - 1, 0, -1):
        last_state = tuple(backpointer[t, last_state[0], last_state[1]])
        best_path.insert(0, last_state)

    return V, best_path, max_prob

# Example usage
input_str = '''4 10
0 0 0 0 X 0 0 0 0 X
X X 0 0 X 0 X X 0 X
X 0 0 0 X 0 X X 0 0
0 0 X 0 0 0 X 0 0 0
4
1011
1010
1000
1100
0.2'''

rows, cols, map_data, num_observations, observations, sensor_error_rate = parse_input(input_str)

# Define the states and observations
states = [(r, c) for r in range(rows) for c in range(cols) if map_data[r][c] == '0']
start_prob = {state: 1 / len(states) for state in states}
trans_prob = compute_transition_probabilities(rows, cols, map_data)
emission_prob = compute_emission_probabilities(rows, cols, sensor_error_rate, map_data)

# Run the Viterbi algorithm
V, best_path, max_prob = viterbi_algorithm(rows, cols, map_data, observations, start_prob, trans_prob, emission_prob)

print("Trellis Matrix V:")
print(V)
print("Most probable path:", best_path)
print("Probability of the path:", max_prob)
