import numpy as np

BIAS = 1 # Bias neuron

def relu(x):
    return np.maximum(0, x)

hidden_act = lambda x: relu(x)
outp_act = lambda x: relu(x)

class HiddenNeuron:
    def __init__(self, weights, qtyIncConnections):
        self.qtyIncConnections = qtyIncConnections
        
        self.weights = weights
    
    def foward(self, x):
        sum = np.dot(x, self.weights)
        return hidden_act(sum)
    
class BiasNeuron:
    def foward(self, x):
        return np.float32(1)
    
class InputNeuron:
    def foward(self, x):
        return x

class HiddenLayer:
    def __init__(self, qtyNeurons):
        pass
    
class InputLayer:
    def __init__(self, qtdNeurons):
        self.qtdNeurons = qtdNeurons
        
        self.neurons: list[InputNeuron] = []
        
        for i in range(self.qtdNeurons - BIAS):
            self.neurons.append(InputNeuron())
        
        self.neurons.append(BiasNeuron())
        
    def foward(self, x):
        res = np.zeros(self.qtdNeurons)
        for i in range(self.qtdNeurons):
            res[i] = self.neurons[i].foward(x)
        
        return res

class ArtificialNeuralNetwork:
    def __init__(self, qtyInputNeuron, qtyHiddenLayers, qtyHiddenNeuronsPerLayer, qtyNeuronsOutput):
        self.qtyInputNeuron = qtyInputNeuron + BIAS
        self.qtyHiddenLayers = qtyHiddenLayers
        self.qtyHiddenNeuronsPerLayer = qtyHiddenNeuronsPerLayer + BIAS
        self.qtyNeuronsOutput = qtyNeuronsOutput
        
        self._create_ann(random=False)
    
    def _create_ann(self, random=True):
        self.inputLayer = InputLayer(self.qtyInputNeuron)
    
    def foward(self, x):
        return self.inputLayer.foward(x)