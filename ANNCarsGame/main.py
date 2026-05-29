import numpy as np

import game
from ann import ArtificialNeuralNetwork, FLOAT_TYPE

INPUT_NEURONS = 2
HIDDEN_LAYERS = 2
HIDDEN_NEURONS_PER_LAYER = 2
OUTPUT_NEURONS = 2

rng = np.random.default_rng(seed=42) 

def main():
    ANN = ArtificialNeuralNetwork(INPUT_NEURONS, HIDDEN_LAYERS, HIDDEN_NEURONS_PER_LAYER, OUTPUT_NEURONS)
    
    x = np.array([1, -1], dtype=FLOAT_TYPE)
    
    print(ANN.foward(x))
    net_arr = ANN.network_to_arr()
    print(net_arr, len(net_arr))
    test_arr = ([rng.uniform(low=-1.5, high=1.5, size=3).dtype(FLOAT_TYPE) for i in range(8)])
    print(test_arr)
    ANN.arr_to_network(test_arr)
    
    print(ANN.foward(x))

if __name__ == "__main__":
    main()