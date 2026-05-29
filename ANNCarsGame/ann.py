import numpy as np
from typing import Literal

BIAS = 1 # Bias neuron
FLOAT_TYPE = np.float32

rng = np.random.default_rng(seed=42)

def relu(x):
    return np.maximum(0, x)

hidden_act = lambda x: relu(x)
outp_act = lambda x: relu(x)

class HiddenNeuron:
    def __init__(self):
        self.weights = 0
        
    def foward(self, x):
        sum = np.dot(x, self.weights)
        return hidden_act(sum)
    
class BiasNeuron:
    def update_weights(self, x): pass
    
    def foward(self, x=None):
        return FLOAT_TYPE(1)
    
class InputNeuron:
    def foward(self, x):
        return x

class HiddenLayer:
    def __init__(self, qtyNeurons, qtyLastLayerNeurons):
        self.qtyNeurons = qtyNeurons
        self.qtyLastLayerNeurons = qtyLastLayerNeurons
        
        self.neurons: list[HiddenNeuron] = []
        
        for i in range(self.qtyNeurons - BIAS):
            self.neurons.append(HiddenNeuron())
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
            self.neurons.append(HiddenNeuron())
    
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
        
        self._create_ann()
        self._set_weights("random")
    
    def _create_ann(self):
        self.inputLayer = InputLayer(self.qtyInputNeuron)
        self.hiddenLayers = [HiddenLayer(self.qtyHiddenNeuronsPerLayer, self.qtyInputNeuron)]
        
        for i in range(1, self.qtyHiddenLayers):
            qtyLastLayerNeurons = self.hiddenLayers[i-1].qtyNeurons
            self.hiddenLayers.append(HiddenLayer(self.qtyHiddenNeuronsPerLayer, qtyLastLayerNeurons))
    
        self.outputLayer = OutputLayer(self.qtyNeuronsOutput, self.hiddenLayers[-1].qtyNeurons)
        
    def _set_weights(self, method: Literal["random", "ones"] = "ones"):
        update_method = {"random":lambda size: rng.uniform(low=-1.5, high=1.5, size=size).astype(FLOAT_TYPE),
                         "ones":lambda size: np.ones(size, dtype=FLOAT_TYPE)}
        
        for i in range(len(self.hiddenLayers)):
            for j in range(len(self.hiddenLayers[i].neurons)):
                size = self.hiddenLayers[i].qtyLastLayerNeurons
                self.hiddenLayers[i].neurons[j].weights = update_method[method](size)
                
        for i in range(len(self.outputLayer.neurons)):
            size = self.outputLayer.qtyLastLayerNeurons
            self.outputLayer.neurons[i].weights = update_method[method](size)
            
    def network_to_arr(self):
        net_arr = []
        
        for layer in self.hiddenLayers:
            for neuron in layer.neurons:
                net_arr.append(neuron.weights)
            
        
        for neuron in self.outputLayer.neurons:
            net_arr.append(neuron.weights)
            
        return net_arr
    
    def arr_to_network(self, net_arr: list):
        for i in range(self.qtyHiddenLayers):
            for j in range(self.qtyHiddenNeuronsPerLayer):
                a = self.hiddenLayers[i].neurons[j].weights = net_arr.pop(0)
                
        for i in range(self.qtyNeuronsOutput):
            self.outputLayer.neurons[i].weights = net_arr.pop(0)
        
    def foward(self, x):
        res_prev = self.inputLayer.foward(x)
        
        for i in range(self.qtyHiddenLayers):
            res_prev = self.hiddenLayers[i].foward(res_prev)
        
        res = self.outputLayer.foward(res_prev)
        
        return res