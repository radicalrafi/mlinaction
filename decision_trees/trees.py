"""
    @file:trees.py
    @desc:implementation of decision trees in python
    @author:rafi
"""

import entropy
import operator

def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,0,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers']
    
    return dataSet,labels


def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if(featVec[axis] == value):
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
"""
now we make a global function that will split 
our dataset and do the entropy calculation needed
on the sub datasets that we splitted then will return
the best feature to split on
"""

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    #calculate the dataset entropy as whole
    baseEntropy = entropy.calcShannonEntropy(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featureList = [example[i] for example in dataSet]
        uniqueVals = set(featureList)
        newEntropy = 0.0
        #now we calculate the entropy for each split
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * entropy.calcShannonEntropy(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCount(classList):
    classCount = {}
    for vote in classList:
        if(vote not in classCount.keys()):
            classCount[vote] = 0
        classCount[vote] += 1

    sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

#generate the decision tree

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    #we stop if all classes are equal
    if(classList.count(classList[0]) == len(classList)):
        return classList[0]
    #when there are no more features we return the majority
    if(len(dataSet[0]) == 1):
            return majorityCount(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree


    
