import numpy as np
import operator

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels=['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sumDiffMat = sqDiffMat.sum(axis=1)
    distance = sumDiffMat**0.5
    sortedDistanceIndicies = distance.argsort()
    classCount = {}
    for i in range(0,k):
        voteLabel = labels[sortedDistanceIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0)+1
    classSorted = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return classSorted[0][0]

def file2matrix(filename):
    f = open(filename)
    arrayLines = f.readlines()
    numberOfLines = len(arrayLines)
    returnMat = np.zeros((numberOfLines,3))
    classLabelVetor = []
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split("\t")
        returnMat[index:] = listFromLine[0:3]
        classLabelVetor.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVetor


