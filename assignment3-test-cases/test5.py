import sys
import numpy as np

def read_input(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    rows, cols = map(int, lines[0].strip().split())
    map_data = [line.strip().split() for line in lines[1:rows+1]]
    num_observations = int(lines[rows+1].strip())
    sensor_observations = [line.strip() for line in lines[rows+2:rows+2+num_observations]]
    error_rate = float(lines[rows+2+num_observations].strip())
    
    return rows, cols, map_data, num_observations, sensor_observations, error_rate

def get_neighbors(i, j, rows, cols):
    neighbors = []
    for x, y in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
        if 0 <= x < rows and 0 <= y < cols:
            neighbors.append((x, y))
    return neighbors

def initialize_transition_matrix(map_data, rows, cols):
    K = rows * cols
    transition_matrix = np.zeros((K, K))
    
    for i in range(rows):
        for j in range(cols):
            if map_data[i][j] == '0':
                neighbors = get_neighbors(i, j, rows, cols)
                for x, y in neighbors:
                    if map_data[x][y] == '0':
                        transition_matrix[i*cols+j, x*cols+y] = 1 / len(neighbors)
    
    return transition_matrix

def initialize_emission_matrix(map_data, sensor_observations, error_rate, rows, cols):
    K = rows * cols
    N = len(sensor_observations)
    emission_matrix = np.zeros((K, N))
    
    for i in range(rows):
        for j in range(cols):
            if map_data[i][j] == '0':
                for obs_idx, obs in enumerate(sensor_observations):
                    if len(obs) != 4:
                        print(f"Debug: sensor_observations[{obs_idx}] = {obs} (length {len(obs)})")
                        raise ValueError(f"Observation is not of length 4 at position ({i}, {j})")
                    
                    correct_bits = sum(map_data[i][j] == '0' for k in range(4) if obs[k] == '1')
                    emission_matrix[i*cols+j, obs_idx] = (1 - error_rate)**correct_bits * error_rate**(4 - correct_bits)
    
    return emission_matrix

def viterbi(rows, cols, initial_prob, transition_matrix, emission_matrix, sensor_observations):
    K = rows * cols
    T = len(sensor_observations)
    trellis = np.zeros((K, T))
    
    for i in range(K):
        trellis[i, 0] = initial_prob[i] * emission_matrix[i, 0]
    
    for j in range(1, T):
        for i in range(K):
            trellis[i, j] = max(trellis[k, j-1] * transition_matrix[k, i] for k in range(K)) * emission_matrix[i, j]
    
    return trellis

def main():
    if len(sys.argv) != 2:
        print("Usage: python viterbi.py [input]")
        return
    
    file_path = sys.argv[1]
    
    rows, cols, map_data, num_observations, sensor_observations, error_rate = read_input(file_path)
    
    transition_matrix = initialize_transition_matrix(map_data, rows, cols)
    emission_matrix = initialize_emission_matrix(map_data, sensor_observations, error_rate, rows, cols)
    
    initial_prob = np.array([1 / (rows * cols) if map_data[i // cols][i % cols] == '0' else 0 for i in range(rows * cols)])
    
    trellis = viterbi(rows, cols, initial_prob, transition_matrix, emission_matrix, sensor_observations)
    
    np.set_printoptions(precision=2, suppress=True)
    print(trellis)
    print(rows, cols, map_data, num_observations, sensor_observations, error_rate)

if __name__ == "__main__":
    main()
