from dataGrabber import dataGrabber
from PEBL import PEBL
from featureSelection import featureSelection
from centroid import Centroid
from helpers import helpers

labelFile = "/home/alex/KnowEng/data/LEE_LIVER_CANCER_ACOX1.CB.txt"
dataFile = "/home/alex/KnowEng/data/ENSG.kegg_pathway.txt"

dataRetriever = dataGrabber()
(featureVectorDict, labelDict, dataIndices) = dataRetriever.getData(labelFile, dataFile, 1, 0)

# features = []
# labels = []
#
# for id in labelDict.iterkeys():
#     features.append(featureVectorDict[id])
#     labels.append(labelDict[id])

#x = featureSelection().rankFeaturesChi2(features, labels)[0:30]
#y = featureSelection().rankFeaturesForest(features, labels, 250)[0:30]
#z = featureSelection().rankFeaturesFourier(features, labels)[0:30]

#print z

# x = PEBL()
# x.train(featureVectorDict, labelDict)

posX = helpers().getFeaturesByLabel(featureVectorDict, labelDict, 1)
unlabX = helpers().getFeaturesByLabel(featureVectorDict, labelDict, 0)

centroidPosX = Centroid().getCentroid(posX)
negIndices = Centroid().getNFarthestPoints(unlabX, centroidPosX, len(posX))

negX = []
for i in negIndices:
    negX.append(unlabX[i])

X = posX + negX
y = [1] * len(posX) + [-1] * len(negX)

print featureSelection().rankFeaturesFourier(X, y)