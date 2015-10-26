from dataGrabber import dataGrabber
from PEBL import PEBL
from featureSelection import featureSelection

labelFile = "/home/alex/KnowEng/data/LEE_LIVER_CANCER_ACOX1.CB.txt"
dataFile = "/home/alex/KnowEng/data/ENSG.kegg_pathway.txt"

dataRetriever = dataGrabber()
(featureVectorDict, labelDict, dataIndices) = dataRetriever.getData(labelFile, dataFile, 1, 0)

features = []
labels = []

for id in labelDict.iterkeys():
    features.append(featureVectorDict[id])
    labels.append(labelDict[id])

x = featureSelection().rankFeaturesChi2(features, labels)[0:30]
y = featureSelection().rankFeaturesForest(features, labels, 250)[0:30]
z = featureSelection().rankFeaturesFourier(features, labels)[0:30]
