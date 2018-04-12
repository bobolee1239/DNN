#!/usr/bin/env python3
#
## ---------- Deep Learning Hw1 -----------
# ID: A061508
#
## ----------------------------------------
import numpy as np
import matplotlib.pyplot as plt

class ActiveFunctions:
    """
    A structure contains various active functions.
    """
    @staticmethod
    def relu(rawValues):
        output = np.zeros(rawValues.shape)
        for idx, value in enumerate(rawValues):
            output[idx] = value if value > 0 else 0
        return output
    @staticmethod
    def softmax(rawValues):
        raise NotImplementedError





## LAYER
class Layer:
    """
    A base class to represent a layer as a list of node and a table of output
    weights.
    """
    stdev = 0.001

    def __init__(self, numberOfInput, numberOfNodes, activeFunc = None):
        """
        Arg:
        ------------------------------------------------------------
        numberOfInputNodes <Int> : number of nodes of last layer.
        numberOfNodes <Int> : number of nodes this layer.
        activeFunc <Function Pointer> : activer function, e.g. softmax, reLu ...
        """
        self.numberOfInput = numberOfInput
        self.numberOfNodes = numberOfNodes
        self.activeFunc = activeFunc if activeFunc else (lambda x : x)
        # randomly initialize parameters
        self.weightTable = np.random.normal(0, Layer.stdev,   \
                                              (numberOfNodes, numberOfInput))
        self.biases = np.random.normal(0, Layer.stdev, (numberOfNodes, ))

    def computeOutput(self, inputData):
        """
        Arg:
        ------------------------------------------------------------
        input: <np.ndarray> a couple of input values in the shape of (self.numberOfInputNodes, ).

        Return:
        ------------------------------------------------------------
        output: <np.ndarray> output values in the shape of (self.numberOfNodes, ).
        """
        # Reserve output memories
        output = self.weightTable.dot(inputData.reshape(self.numberOfInput, 1)) + self.biases.reshape(self.numberOfNodes, 1)
        return self.activeFunc(output.ravel())

## DNN
class DNN:
    """
    < Deep Neuro Network >
    """
    def __init__(self, learningRate, costFunction):
        """
        learningRate <Float> : learning rate to do SGD.
        costFunction <String> : lower case string. e.g. LMS, cross-entropy
        """
        self.layers = []
        self.lr = learningRate
        self.costFunction = costFunction.lower()
    def add(self, numberOfNodes, numberOfInput = None, activeFunc = None):
        """
        Add a new layer into Deep Neuro Network
        """
        if not self.layers:
            # the first layer
            self.layers.append(Layer(numberOfInput, numberOfNodes, activeFunc))
        else:
            self.layers.append(Layer(self.layers[-1].numberOfNodes, numberOfNodes, activeFunc))
    def forwardPropagation(self, inputData):
        propagationInput = inputData
        for layer in self.layers:
            propagationInput = layer.computeOutput(propagationInput)
        return propagationInput
    def backPropagation(self):
