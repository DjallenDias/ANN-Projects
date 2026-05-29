import numpy as np

import game, ann

def main():
    ANN = ann.ArtificialNeuralNetwork(2, 2, 2, 2)
    
    x = np.array([1, -1], dtype=np.float32)
    
    print(ANN.foward(x))

if __name__ == "__main__":
    main()