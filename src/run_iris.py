from PEBL import PEBL
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

print y

for i in range(0, len(y)):
    if y[i] != 0:
        y[i] = 1

featureDict = {}
labelDict = {}
for i in range(10, len(y)):
    id = str(i)
    featureDict[id] = X[i]
    if y[i] == 0:
        labelDict[id] = 0
    else:
        labelDict[id] = 1

classifierPEBL = PEBL()
trainedSVM = classifierPEBL.train(featureDict, labelDict)

print trainedSVM.predict(X[0:10]), y[0:10]