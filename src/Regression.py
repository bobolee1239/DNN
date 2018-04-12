
# coding: utf-8

# In[16]:


#!/usr/bin/env python3


# ### Deep Learning HW1
# ---
# #### Problem1
# - ID: A061508
# - NAME: 李宗翰
# ---

# In[17]:


from DNN_Jupyter import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[ ]:


FILENAME = '../Dataset/energy_efficiency_data.csv'
NUM_TRAIN = 576


# In[ ]:


def readFile(filename):
    """
    Returns:
    ------------------------------
    1. rawDatas: <list of numpy.ndarray> shape = (#samples, #features).
    2. targets: <numpy.ndarray> shape = (#samples, ) corresponds to datas.
    3. datas: <list of numpy.ndarray> normalize rawDatas with infinity norm.
    
    """
    df = pd.read_csv(filename)

    indexOfOrientation = 5
    indexOfGlazingAreaDistribution = 7
    indexOfTarget = 8
    
    rawDatas = [] # list to store features
    maxValues = [0.0] * 6 # dictionary<Int, Float> establish table to check out infinite norm
    targets = []
    
    for rawData in df.values:
        sample = []
        
        ## Dealing with Normal Vector
        for idx in (0, 1, 2, 3, 4, 6):
            sample.append(rawData[idx])
            for idx, var in enumerate(sample):
                if var > maxValues[idx]: maxValues[idx] = var
                   
        ## Dealing with One Hot Vector
        #  1. Orientation
        indexOfOne = (rawData[indexOfOrientation] - 1) % 4
        for i in range(4):
            if i == indexOfOne: sample.append(1)
            else: sample.append(0)
                
        #  2.Glazing Area Distribution
        indexOfOne = (rawData[indexOfGlazingAreaDistribution] % 6) - 1
        for i in range(5):
            if i == indexOfOne: sample.append(1)
            else: sample.append(0)
        
        rawDatas.append(np.array(sample))
        
        targets.append(rawData[indexOfTarget])
     
    normalizedDatas = []
    ## Normalize Data
    for sample in rawDatas:
        tmp = [0.0] * 15
        for idx in range(len(maxValues)):
            tmp[idx] = sample[idx] / maxValues[idx]
        for idx in range(6, 15):
            tmp[idx] = sample[idx]
        normalizedDatas.append(np.array(tmp))
    
    return (rawDatas, normalizedDatas, np.array(targets))
    
    
    


# In[ ]:


if __name__ =='__main__':
    rawDatas, normalizedDatas, targets = readFile(FILENAME)
    dataSet = list(zip(normalizedDatas, targets))
    
    np.random.shuffle(dataSet)
    trainingSet = dataSet[:NUM_TRAIN]
    testingSet = dataSet[len(dataSet)-NUM_TRAIN:]
    
    dnn = DNN(learningRate = 0.001, costFunction = 'LMS')
    dnn.add(numberOfInput = 15, numberOfNodes = 10, activeFunc = ReLu())
    dnn.add(numberOfNodes = 10, activeFunc = ReLu())
    dnn.add(numberOfNodes = 1, activeFunc = ActiveFunctions())

    dnn.fit(trainingSet, toPlot=True, toPrint=True, maxEpochs=1000)


# In[ ]:


dnn.predict(testingSet, toPlot=True)
plt.show()


# In[ ]:


# ## Testing

# df = pd.read_csv(FILENAME)
# for value in df.values:
#     print(value)
    
    
# #     print('RAW\n---------------------------------------')
# #     for d in rawDatas:
# #         print(d)
        
# #     print('NORMALIZE\n---------------------------------')
# #     for d in normalizedDatas:
# #         print(d)
    
#     print('TARGET\n-------------------------------------')
#     for t in targets:
#         print(t)

