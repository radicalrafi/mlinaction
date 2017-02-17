"""
Support Vector Machines from Scratch
Using Platt 'Sequential Minimal Optimisation"
@author:rafi
@ref:MLiAction
"""
from numpy import *
import random

#helper function for simple SMO implementation
def load_dataset(filename):
    """
    loads a dataset from a file onto a data matrix and label vector
    Args:
        filename
    Return:
        data matrix,label vector
    """
    dataMat = []
    labelMat = []
    fp = open(filename)
    for line in fp.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def select_random(i,m):
    """
    select a random integer from a range
    i : index of first alpha
    m : number of alphas

    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

def clip_alpha(aj, H, L):
    """
    clip values if they get too big
    Args:
    alpha,high,low
    """
    if aj > H:
        aj = H
    elif aj < L:
        aj = L
    return aj

def simple_smo(dataMatIn, classLabels, C, toler, maxIter):
    """
    in each iteration < maxIter we set alphaPairsChanged to Zero and go trough 
    the entire set sequentially ,alphaPairsChanged is used to store number of alphas we optimised
    fXi: prediction label
    Ei: error calculated based on the prediction vs the true class label
    if Ei is large the alpha corresponding to this instance can be optimised
    Alphas are cliped to 0 and C making them bounded so if any alpha is equal to 0 or C it's not worth to optimise
    We then select a random alpha we calculate it's Error (fXj,Ej) we keep copies of both alphas to compare them
    to the optimised alphas to see how everything is going on 
    """
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = select_random(i,m)
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print "L==H"; continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print "eta>=0"; continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clip_alpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print "iteration number: %d" % iter
    return b,alphas