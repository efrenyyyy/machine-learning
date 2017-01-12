import kNN
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

group, labels = kNN.createDataSet()
#print kNN.classify0([0,1], group, labels, 3)
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
#print datingDataMat
#print array(datingLabels)
normDataSet, ranges, minValues = kNN.autoNorm(datingDataMat)
#print normDataSet
#print ranges
#print minValues
kNN.datingClassTest()
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))
#plt.show()