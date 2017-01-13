import numpy as np
import operator
from os import listdir

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

def autoNorm(dataSet):
    minValues = dataSet.min(0)
    maxValues = dataSet.max(0)
    ranges = maxValues - minValues
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minValues, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))
    return normDataSet, ranges, minValues,

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minValues = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVec = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVec):
        classifierResult = classify0(normMat[i,:], normMat[numTestVec:m, :], datingLabels[numTestVec:m], 3)
        print "the classifier came back with %d, the real answer is %d" %(classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]):
            errorCount += 1
    print "the total error rate is %f" %(errorCount/float(numTestVec))

def classifyPersion():
    resultList = ["not at all", "in small doses", "in large doses"]
    percetTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    normMat, ranges, minValues = autoNorm(datingDataMat)
    inArr = np.array([percetTats, ffMiles, iceCream])
    classifyResult = classify0((inArr-minValues)/ranges, normMat, datingLabels, 3)
    print "You will probably like this person: ", resultList[classifyResult]

def img2vector(filename):
    returnVect = np.zeros((1,1024))
    f = open(filename)
    for i in range(32):
        listStr = f.readline()
        for j in range(32):
            returnVect[0, i*32+j] = listStr[j]
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' %fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with %d, the real answer is %d" %(classifierResult, classNumStr)
        if (classifierResult != classNumStr):
            errorCount += 1
    print "the total number of errors is %d" %errorCount
    print "the total error rate is %f" %(errorCount/float(mTest))
