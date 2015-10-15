from sklearn import svm
from dataGrabber import dataGrabber

labelFile = "/home/alex/KnowEng/data/LEE_LIVER_CANCER_ACOX1.CB.txt"
dataFile = "/home/alex/KnowEng/data/ENSG.kegg_pathway.txt"

dataRetriever = dataGrabber()
(positiveData, unlabeledData) = dataRetriever.getData(labelFile, dataFile, 1, 0)

# print len(positiveData.keys())
# print len(unlabeledData.keys())

data = []
labels = []

for positiveFeature in positiveData.itervalues():
    data.append(positiveFeature)
    labels.append(1)

for negativeFeature in unlabeledData.itervalues():
    data.append(negativeFeature)
    labels.append(0)
