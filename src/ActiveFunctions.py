
# coding: utf-8

# In[32]:


#!/usr/bin/env python3

import numpy as np


# In[33]:


class ActiveFunctions:
    """
    A base class of activation functions.
    """
    @staticmethod
    def at(rawValues):
        return rawValues
    @staticmethod
    def gradientAt(rawValues):
        raise 1.0


# In[34]:


class ReLu(ActiveFunctions):
    """
    ReLu activation function.
    """
    @staticmethod
    def at(rawValues):
        if type(rawValues) != np.ndarray:
            return rawValues if rawValues > 0.0 else 0.0
        
        output = np.zeros(rawValues.shape)
        for idx, value in enumerate(rawValues):
            output[idx] = value if value > 0.0 else 0.0
        return output
    
    @staticmethod
    def gradientAt(rawValues):
        if type(rawValues) != np.ndarray:
            return 1.0 if rawValues > 0.0 else 0.0
        
        output = np.zeros(rawValues.shape)
        for idx, value in enumerate(rawValues):
            output[idx] = 1.0 if value > 0.0 else 0.0
        return output


# In[37]:


## --------- TESTING --------

if __name__ =='__main__':
    # --- Testing ReLu:
    relu = ReLu()
    testData = np.array([1.0, 100.0, -50.0, 7.56])
    print(relu.at(testData))
    print(relu.gradientAt(testData))
    
    # --- Testing Softmax:

