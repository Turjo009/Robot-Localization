import numpy as np
import sys

def parse_input(input_str):
    lines = input_str.split('\n')
    rows = int(lines[0][0])
    columns = int(lines[0][-1])
    map_data = []
    for i in range(1,rows+1):
        map_data.append(lines[i].strip(' '))
    no_of_observations = lines[rows+1]
    observation_list = lines[rows+2:-1]
    error_rate = lines[-1]

    return rows, columns, map_data, no_of_observations, observation_list, error_rate
    # print(rows, '|', columns,'|', map_data,'|', no_of_observations, '|',observation_list,'|', error_rate)


def binary_to_decimal(given_list):
    decimal_numbers = [int(binary_str, 2) + 1 for binary_str in given_list]
    return decimal_numbers


def observation_space(no_of_observations):
    max_value = 2 ** no_of_observations 

    binary_strings_list = []
    for num in range(max_value):
        binary_str = format(num, f'0{no_of_observations}b')
        binary_strings_list.append(binary_str)

    return binary_strings_list


def observation_space_to_list(observation_space_binary_strings):
    observation_space_list = []
    for i, binary_str in enumerate(observation_space_binary_strings):
        observation = [int(bit) for bit in binary_str]
        observation_space_list.append({i : observation})
    return observation_space_list


def state_space(map_data):
    state_space_list = []
    for x, line in enumerate(map_data):
        index_list = line.split(' ')
        for y, index in enumerate(index_list):
            if index == '0':
                state_space_list.append((x,y))
    return state_space_list


def transmission_matrix(map_data, cols):
    # S is the coordinates of the 'O's in the map
    S = state_space(map_data)
    K = len(S)
    # print('K', K)
    # print('S', S)
    rows = len(map_data)
    map_data = [row.split() for row in map_data]

    transition_matrix = np.zeros((K, K))

    def get_valid_neighbors(x, y):
        potential_neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return [
            (nx, ny) for nx, ny in potential_neighbors
            if 0 <= nx < rows and 0 <= ny < cols and map_data[nx][ny] == '0'
        ]

    for i, (x, y) in enumerate(S):
        valid_neighbors = get_valid_neighbors(x, y)
        num_valid_neighbors = len(valid_neighbors)
        for nx, ny in valid_neighbors:
            j = S.index((nx, ny))
            transition_matrix[i, j] = 1 / num_valid_neighbors
    
    return transition_matrix

    # print("Transition matrix:", transition_matrix)


def actual_observation(map_data):
    rows = len(map_data)
    cols = len(map_data[0].split())
    map_data = [row.split() for row in map_data]
    observation_list = []

    for x in range(rows):
        for y in range(cols):
            if map_data[x][y] == '0':
                # Check North
                north = 1 if x == 0 or map_data[x-1][y] == 'X' else 0
                # Check South
                south = 1 if x == rows-1 or map_data[x+1][y] == 'X' else 0
                # Check West
                west = 1 if y == 0 or map_data[x][y-1] == 'X' else 0
                # Check East
                east = 1 if y == cols-1 or map_data[x][y+1] == 'X' else 0
                
                # Combine results into a list
                observation = [north, south, west, east]
                observation_list.append(observation)

    return observation_list


def count_differences(list1, list2):
    # Count the number of differing elements
    count = sum(el1 != el2 for el1, el2 in zip(list1, list2))
    return count


def emission_matrix(map_data,error_rate):

    # S is the coordinates of the 'O's in the map
    S = state_space(map_data)
    # K is the number of 'O's in the map
    K = len(S)
    # print('K', K)
    # print('S', S)
    # since we have only 4 observations, therefore 2**4 = 16
    N = 16 
    Em = np.zeros((K, N))

    for i in range(1, K+1):
        actual_obs = actual_observation(map_data) # [N, S, W, E]
        # print("actual observations: ", actual_obs)
        observation_list = observation_space_to_list(observation_space(4))
        # print('O', observation_list)
        
        for j in range(1, N+1):
            observation = observation_list[j-1] #[N, S, W, E]
            count = count_differences(actual_obs[i-1], observation[j-1])
            # print("actual observations: ", actual_obs[i-1])
            # print("Obserbation", observation)
            # print("count", count)
            Em[i-1][j-1] = (1-error_rate)**(4 - count) * error_rate**count
    return Em

    # print("EMIssion matrix:", Em)
    # print('End')



def viterbi_forward(map_data, Y, Tm, Em):
    S = state_space(map_data)
    K = len(S)
    T = len(Y)
    initial_probability = [1/K] * K 

    trellis = np.zeros((K, T))
    for i in range(K):
        trellis[i, 0] = initial_probability[i] * Em[i][Y[0] -1]

    for j in range(1,T):
        for i in range(K):
            given_formula = [trellis[k, j-1] * Tm[k, i] * Em[i][Y[j]-1] for k in range(K)]
            max_probability = max(given_formula)
            trellis[i, j] = max_probability 

    return trellis


def prepare_output(rows, cols, mapdata, trellis):
    map_size = [rows,cols]
    result = [np.zeros(map_size) for i in range(len(Y))]
    S = state_space(mapdata)
    transposed_trellis = np.transpose(trellis)
    for t, prob in enumerate(transposed_trellis):
        for index, p in enumerate(prob):
            i,j = S[index]
            result[t][i][j] = p
    return result

    

def read_input_from_file(file_name):
    with open(file_name, 'r') as file:
        input_str = file.read()
    return input_str


if __name__ == "__main__":
    # Get input file name from command-line argument
    if len(sys.argv) != 2:
        print("Usage: python file.py [inputfile]")
        sys.exit(1)
    
    input_file = sys.argv[1]

    # Read content from file
    input_str = read_input_from_file(input_file)

    # print(parse_input(input_str))
    rows, columns, map_data, no_of_observations, observation_list, error_rate = parse_input(input_str)
    # print(observation_space(4))
    # print(state_space(map_data))

    Em = emission_matrix(map_data,float(error_rate))
    Tm = transmission_matrix(map_data, columns)
    # print(Em,Tm)
    Y = binary_to_decimal(observation_list)
    # print(Y)


    trellis = viterbi_forward(map_data, Y, Tm, Em)
    # print(trellis)

    final_result = prepare_output(rows, columns, map_data, trellis)


    # print(final_result)
    np.savez("output.npz", *final_result)
