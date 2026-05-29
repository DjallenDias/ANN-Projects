import numpy as np

BIAS = 1 # Bias neuron

def relu(x):
    return np.maximum(0, x)

hidden_act = lambda x: relu(x)
outp_act = lambda x: relu(x)

class HiddenNeuron:
    def __init__(self, weights):
        self.weights = weights
    
    def foward(self, x):
        sum = np.dot(x, self.weights)
        return hidden_act(sum)
    
class BiasNeuron:
    def foward(self, x=None):
        return np.float32(1)
    
class InputNeuron:
    def foward(self, x):
        return x

class HiddenLayer:
    def __init__(self, qtyNeurons, qtyLastLayerNeurons):
        self.qtyNeurons = qtyNeurons
        self.qtyLastLayerNeurons = qtyLastLayerNeurons
        
        self.neurons: list[HiddenNeuron] = []
        
        for i in range(self.qtyNeurons - BIAS):
            self.neurons.append(HiddenNeuron(np.ones(qtyLastLayerNeurons)))
        self.neurons.append(BiasNeuron())
    
    def foward(self, x):
        res = np.zeros(self.qtyNeurons)
        for i in range(self.qtyNeurons):
            res[i] = self.neurons[i].foward(x)
        
        return res
    
class InputLayer:
    def __init__(self, qtdNeurons):
        self.qtdNeurons = qtdNeurons
        
        self.neurons: list[InputNeuron] = []
        
        for i in range(self.qtdNeurons - BIAS):
            self.neurons.append(InputNeuron())
        
        self.neurons.append(BiasNeuron())
        
    def foward(self, x):
        res = np.zeros(self.qtdNeurons)
        for i in range(self.qtdNeurons - BIAS):
            res[i] = self.neurons[i].foward(x[i])
            
        res[-1] = self.neurons[-1].foward()
        
        return res

class OutputLayer:
    def __init__(self, qtyNeurons, qtyLastLayerNeurons):
        self.qtyNeurons = qtyNeurons
        self.qtyLastLayerNeurons = qtyLastLayerNeurons
        
        self.neurons: list[HiddenNeuron] = []
        
        for i in range(self.qtyNeurons):
            self.neurons.append(HiddenNeuron(np.ones(qtyLastLayerNeurons)))
    
    def foward(self, x):
        res = np.zeros(self.qtyNeurons)
        for i in range(self.qtyNeurons):
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
        self.hiddenLayers = [HiddenLayer(self.qtyHiddenNeuronsPerLayer, self.qtyInputNeuron)]
        
        for i in range(1, self.qtyHiddenLayers):
            qtyLastLayerNeurons = self.hiddenLayers[i-1].qtyNeurons
            self.hiddenLayers.append(HiddenLayer(self.qtyHiddenNeuronsPerLayer, qtyLastLayerNeurons))
    
        self.outputLayer = OutputLayer(self.qtyNeuronsOutput, self.hiddenLayers[-1].qtyNeurons)
        
    def foward(self, x):
        res_prev = self.inputLayer.foward(x)
        
        for i in range(self.qtyHiddenLayers):
            res_prev = self.hiddenLayers[i].foward(res_prev)
        
        res = self.outputLayer.foward(res_prev)
        
        return res