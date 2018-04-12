
# coding: utf-8

# In[ ]:


##### !/usr/bin/env python3
#
## ---------- Deep Learning Hw1 -----------
# ID: A061508
#
## ----------------------------------------


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
# import ActiveFunctions


# In[ ]:


class ActiveFunctions:
    """
    A base class of activation functions.
    """
    def at(self, rawValues):
        return rawValues
    
    def gradientAt(self, rawValues):
        return np.array([1.0])

class ReLu(ActiveFunctions):
    """
    ReLu activation function.
    """
    def at(self, rawValues):
        if type(rawValues) != np.ndarray:
            return rawValues if rawValues > 0.0 else 0.0
        
        output = np.zeros(rawValues.shape)
        for idx, value in enumerate(rawValues):
            output[idx] = value if value > 0.0 else 0.0
        return output
    
    def gradientAt(self, rawValues):
        if type(rawValues) != np.ndarray:
            return 1.0 if rawValues > 0.0 else 0.0
        
        output = np.zeros(rawValues.shape)
        for idx, value in enumerate(rawValues):
            output[idx] = 1.0 if value > 0.0 else 0.0
        return output


# In[ ]:


## Fully Connected LAYER

class Layer:
    """
    A base class to represent a layer as a list of node and a table of output
    weights.
    """
    stdev = 0.01

    def __init__(self, numberOfInput, numberOfNodes, activeFunc = ActiveFunctions()):
        """
        Arg:
        ------------------------------------------------------------
        numberOfInputNodes <Int> : number of nodes of last layer.
        numberOfNodes <Int> : number of nodes this layer.
        activeFunc <Function Pointer> : activer function, e.g. softmax, reLu ...
        """
        self.numberOfInput = numberOfInput
        self.numberOfNodes = numberOfNodes
        self.activeFunc = activeFunc
        self.rawOutputs = []
        self.outputs = []
        
        
        # randomly initialize parameters
        self.weightTable = np.random.normal(0, Layer.stdev,                                                 (numberOfNodes, numberOfInput))
        self.biases = np.random.normal(0, Layer.stdev, (numberOfNodes, ))
        
        # Reserve memory for gradients
        self.gradientOfBiases = np.zeros(self.biases.shape)
        self.gradientOfWeightTable = np.zeros(self.weightTable.shape)

    def computeOutput(self, inputData, toUpdate=False):
        """
        Arg:
        ------------------------------------------------------------
        input: <np.ndarray> a couple of input values in the shape of (self.numberOfInputNodes, ).

        Return:
        ------------------------------------------------------------
        output: <np.ndarray> output values in the shape of (self.numberOfNodes, ).
        """
        # Reserve output memories
        rawOutput = self.weightTable.dot(inputData.reshape(self.numberOfInput, 1)) + self.biases.reshape(self.numberOfNodes, 1)
        output = self.activeFunc.at(rawOutput.ravel())
        
        # Store RawOutput ans Output
        if toUpdate:
            self.rawOutputs.append(rawOutput.ravel())
            self.outputs.append(output)
        
        return output
    
    def computeOutputs(self, inputDatas, toUpdate = False):
        results = []
        for data in inputDatas:
            results.append(self.computeOutput(data, toUpdate))
        return results
    
    def update(self, preGradients, lastLayerOutput, lr, toPrint = False):
        """
        Arg:
        --------------------------
        - preGradients <list of np.ndarray> : store multiple gradients from next layer.
        
        """
        assert type(preGradients[0]) == np.ndarray, '<Usage> AnLayer.update(preGradients) where preGradients should be an numpy.ndarray'
        
        gradients = []      # store gradients to propagation
        gradientsOfBiases = []
        gradientsOfWeightTable = []
        
        for idx, preGradient in enumerate(preGradients):
            preGradient = preGradient.reshape(len(preGradient), 1)
            tmpGradient = preGradient * self.activeFunc.gradientAt(self.rawOutputs[idx]).reshape(len(preGradient), 1)
            tmpGradientOfBiases = tmpGradient #+ 0.01*(self.biases.reshape(len(tmpGradient), 1)**2)
            
            if toPrint:
                print('tmpGradient:', tmpGradient)
                print('preGradient:', preGradient)
                print('activeFunc:', self.activeFunc.gradientAt(self.rawOutputs[idx]))
                print('index:', idx)
                print('----------------')
                
            tmpGradientOfWeightTable = tmpGradient.reshape(len(tmpGradient), 1).dot(
                lastLayerOutput[idx].reshape(1, len(lastLayerOutput[idx]))) #+ 0.01*(self.weightTable**2)
            
            gradient = self.weightTable.T.dot(tmpGradient.reshape(len(tmpGradient), 1))
            
            # store values
            gradients.append(gradient)
            gradientsOfBiases.append(tmpGradientOfBiases)
            gradientsOfWeightTable.append(tmpGradientOfWeightTable)
            
        # compute stochastic gradient
        sgOfBiases = sum(gradientsOfBiases) / float(len(gradientsOfBiases))
        sgOfWeightTable = sum(gradientsOfWeightTable) / float(len(gradientsOfWeightTable))
        
        if toPrint:
            print('sgOfBiases:', lr * sgOfBiases)
            print('sgOfWeightTable', lr * sgOfWeightTable)
        
        # update parameters
        self.biases -= lr * sgOfBiases.reshape(len(self.biases))
        self.weightTable -= lr * sgOfWeightTable
        
        # reset rawOutputs
        self.rawOutputs = []
        self.outputs = []
        
        return gradients
        


# In[1]:


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
        self.regularizer = None
    def add(self, numberOfNodes, numberOfInput = None, activeFunc = ActiveFunctions()):
        """
        Add a new layer into Deep Neuro Network
        """
        if not self.layers:
            # the first layer
            self.layers.append(Layer(numberOfInput, numberOfNodes, activeFunc))
        else:
            self.layers.append(Layer(self.layers[-1].numberOfNodes, numberOfNodes, activeFunc))
            
    def forwardPropagation(self, inputDatas, toUpdate = False):
        propagationInputs = inputDatas
        
        for layer in self.layers:
            propagationInputs = layer.computeOutputs(propagationInputs, toUpdate)
            
        return propagationInputs
    def backPropagation(self, miniBatch, toPrint = False):
        """
        Arg:
        ------------------------
        miniBatch <tuple> (datas, targets)
        datas <list of numpy.ndarray> : a list storing input datas
        targets <list of numpy.ndarray> : a list storing targets correspond to datas
        """
        datas, targets = miniBatch
        # forward propagation
        outputs = self.forwardPropagation(datas, toUpdate = True)
        
        preGradients = []
        # take gradient to output
        if self.costFunction == 'lms':
            for t, p in zip(targets, outputs):
                preGradients.append(-2*(t - p))
            preGradients = np.array(preGradients)
        
        elif self.costFunction == 'cross-entropy':
            raise NotImplementedError
            
        else:
            assert False, "No Support for {} cost function".format(self.costFunction)
            
        for idx in reversed(range(len(self.layers))):
            if toPrint: print('Propagation to #{} layer'.format(idx))
                
            if idx > 0:
                preGradients = self.layers[idx].update(preGradients, self.layers[idx - 1].outputs, self.lr)
            else:
                preGradients = self.layers[idx].update(preGradients, datas, self.lr)
            
            if toPrint: print('Pregradints:', preGradients)
                
    def fit(self, trainingSet, stopCriterion=0.001, maxEpochs=30000, miniBatchSize=64, toPlot=False, toPrint=False):
        """
        Arg:
        ---------------------
        1. trainingSet: (trainingDatas, trainingTargets)
            * trainingTargets: <numpy.ndarray>, shape = (#samples, dimOfTargets)
            * trainingDatas: <numpy.ndarray>, shape=(#samples, #features) 
        2. stopCriterion: <float>
        3. maxEpochs: <Int>
        
        """
        DATA = 0
        TARGET = 1
        numMiniBatches = len(trainingSet) // miniBatchSize
        errs = []
        err = 1.0
        numEpochs = 0
        
        while (numEpochs < maxEpochs): #and (err > stopCriterion):
            numEpochs += 1
            if toPrint: print('Learning Epoch #{} ...'.format(numEpochs))
            
            # shuffle training datas for partition mini batch
            np.random.shuffle(trainingSet)
            
            # Retrieve Data and Target
            datas = [x[DATA] for x in trainingSet]
            datas = np.array(datas)
            targets = [x[TARGET] for x in trainingSet]
            targets = np.array(targets)
            
            for i in range(numMiniBatches):
                # Slicing
                miniBatchDatas = datas[i*miniBatchSize:(i+1)*miniBatchSize]
                miniBatchTargets = targets[i*miniBatchSize:(i+1)*miniBatchSize]
                
                self.backPropagation((miniBatchDatas, miniBatchTargets))
            
            if self.costFunction == 'lms':
                predict = np.array(self.forwardPropagation(datas))
                err = predict.reshape(targets.shape) - targets
                err = sum(err ** 2) / float(len(targets))
                errs.append(err)
                err = np.linalg.norm(err) 
        
        if toPlot:
            plt.figure()
            plt.plot(list(range(1, numEpochs + 1)), errs, label='Error History')
            plt.xlabel('# Epochs')
            plt.ylabel('RMS Error')
            plt.title('Training History')
      
    def evaluate(self, testingSet):
        """
        Arg:
        ---------------------
        1. testingSet: (trainingDatas, trainingTargets)
            * testingTargets: <numpy.ndarray>, shape = (#samples, dimOfTargets)
            * testingDatas: <numpy.ndarray>, shape=(#samples, #features) 
        
        """
        DATA = 0
        TARGET = 1
        
        datas = [x[DATA] for x in testingSet]
        datas = np.array(datas)
        targets = [x[TARGET] for x in testingSet]
        targets = np.array(targets)
        
        results = np.array(self.forwardPropagation(datas))
        err = results.reshape(targets.shape) - targets
        err = sum(err ** 2) / float(len(err))
        
        return err
    
    def predict(self, testingSet, toPlot = False):
        """
        Arg:
        ---------------------
        1. testingSet: (trainingDatas, trainingTargets)
            * testingTargets: <numpy.ndarray>, shape = (#samples, dimOfTargets)
            * testingDatas: <numpy.ndarray>, shape=(#samples, #features) 
        
        """
        DATA = 0
        TARGET = 1
        
        datas = [x[DATA] for x in testingSet]
        datas = np.array(datas)
        targets = [x[TARGET] for x in testingSet]
        targets = np.array(targets)
        
        results = np.array(self.forwardPropagation(datas))
        
        if toPlot:
            x_axis = list(range(1, len(targets) + 1))
            plt.figure()
            plt.plot(x_axis, targets, label = 'Target')
            plt.plot(x_axis, results, label = 'Prediction')
            plt.xlabel('case No.')
            plt.ylabel('heat load')
            plt.legend()
            plt.title('heating load prediction')
        
        return results
                


# In[ ]:


if __name__ == '__main__':
    dnn = DNN(0.001, 'LMS')
    dnn.add(numberOfInput = 5, numberOfNodes = 10, activeFunc = ReLu())
    dnn.add(numberOfNodes = 3, activeFunc = ReLu())
    dnn.add(numberOfNodes = 1, activeFunc = ActiveFunctions())
    datas = [np.array([1, 2, 3, 4, 5]), np.array([1, 0, 0, 5, 6])]
    #print(dnn.forwardPropagation(datas))
    
    targets = [1, 10]
    dnn.fit(list(zip(datas, targets)), miniBatchSize = 1, toPlot = True)
    
#     for i in range(10000):
#         dnn.backPropagation((datas, targets))
    
    print(dnn.forwardPropagation(datas))
    print(dnn.predict(list(zip(datas, targets))))
#     count = 0
#     for layer in dnn.layers:
#         if layer.rawOutputs != []: count += 1
#         if layer.outputs != []: count += 1
#     print('count:', count)
            
#     print(dnn.layers[0].rawOutputs)
#     print(dnn.layers[1].outputs)
    
#     print((dnn.layers[0].weightTable.dot(np.array([1, 2, 3, 4, 5])) + dnn.layers[0].biases))


# In[ ]:


plt.show()

