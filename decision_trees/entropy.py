"""
@file:entropy.py
@desc:calculating the shannon entropy of a dataset 
@author:rafi
"""

from math import log

def calcShannonEntropy(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    #create a dictionnary of each feature vector
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if(currentLabel not in labelCounts.keys()):
            labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
    
    #calculations
    entropy = 0.0
    for key in labelCounts :
        prob = float(labelCounts[key])/numEntries
        entropy -= prob * log(prob,2)
    return entropy

