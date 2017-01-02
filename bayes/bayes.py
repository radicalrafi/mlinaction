import numpy as np


def load_dataset():
    postingList = [['my','dog','has','flea','problems','please','help'],
                    ['maybe','not','take','him','to','dog','park','stupid'],
                    ['my','dalmation','is','so','cute','I','love','him'],
                    ['stop','posting','stupid','worthless','garbage'],
                    ['mr','licks','ate','my','meat','steack','how','to','stop',
                        'him'],
                    ['maybe','stop','buying','worthless','dog','food','stupid']]    
    classVec = [0,1,0,1,0,1]
    return postingList, classVec

def create_vocab_list(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def words2vec(vocabList, inputSet):
    returnVec = [0]*(len(vocabList))
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "word %s is not in the vocabulary" % word
    return returnVec

def train_bayes_classifier(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    #get probabilities of each category
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0num = zeros(numWords)
    p1num = zeros(numWords)
    p0denom = 0.0
    p1denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1num += trainMatrix[i]
            p1denom += sum(trainMatrix[i])
        else:
            p0num += trianMatrix[i]
            p0denom += sum(trainMatrix[i])

    p1Vec = p1num/p1denom
    p0Vec = p0num/p0denom

    return p0Vec,p1Vec,pAbusive
