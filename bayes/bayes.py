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

def bagOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*(len(vocabList))
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def bayes_classify(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)

    if p1 > p0:
        return 1
    else :
        return 0

def test_bayes():
    listOfPosts,listClasses = load_dataset()
    myVocabList = create_vocab_list(listOfPosts)
    trainMat = []
    for post in listOfPosts:
        trainMat.append(words2vec(myVocabList,post))
    p0,p1,pAb = train_bayes_classifier(np.array(trainMat),np.array(listClasses))
    test1 = ['i','love','my','dalmatian']
    test2 = ['i','hate','you','stupid']
    doc1 = np.array(words2vec(myVocabList,test1))
    doc2 = np.array(words2vec(myVocabList,test2))
    print test1, " as ",bayes_classify(doc1,p0,p1,pAb)
    print test2, " as ",bayes_classify(doc2,p0,p1,pAb)


def train_bayes_classifier(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    #get probabilities of each category
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0num = np.zeros(numWords)
    p1num = np.zeros(numWords)
    p0denom = 0.0
    p1denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1num += trainMatrix[i]
            p1denom += sum(trainMatrix[i])
        else:
            p0num += trainMatrix[i]
            p0denom += sum(trainMatrix[i])

    p1Vec = p1num/p1denom
    p0Vec = p0num/p0denom

    return p0Vec,p1Vec,pAbusive

#spam test lines
