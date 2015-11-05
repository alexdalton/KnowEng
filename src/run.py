from dataGrabber import dataGrabber, dataFileDescriptor
from PEBL import PEBL
from featureSelection import featureSelection
from centroid import Centroid
from helpers import helpers

#labelFile = "/home/alex/KnowEng/data/LEE_LIVER_CANCER_ACOX1.CB.txt"
breastCancerLabelFile = "/home/alex/KnowEng/data/VANTVEER_BREAST_CANCER_ESR1.CB.txt"
keggDataFile = dataFileDescriptor("/home/alex/KnowEng/data/ENSG.kegg_pathway.txt")
goDataFile = dataFileDescriptor("/home/alex/KnowEng/data/ENSG.go_%_evid.txt",
                                lambda x: x >= 40 and x <= 2000)


dataRetriever = dataGrabber()
(featureVectorDict, labelDict, dataIndices) = dataRetriever.getData(breastCancerLabelFile, [goDataFile])

#dataRetriever.convertToCSV(featureVectorDict, labelDict, "BC_GO_FILT_RIPPER.csv")

# newFeatureDict = {}
# newLabelDict = {}
# posCount = 0
# unlabeledCount = 0
#
# for k, v in featureVectorDict2.iteritems():
#     if k in featureVectorDict1:
#         newFeatureDict[k] = v + featureVectorDict1[k]
#         newLabelDict[k] = labelDict1[k]
#         if labelDict1[k] == 1:
#             posCount += 1
#         else:
#             unlabeledCount += 1


# posX = helpers().getFeaturesByLabel(featureVectorDict, labelDict, 1)
unlabX = helpers().getFeaturesByLabel(featureVectorDict, labelDict, 0)
#
#
#
# centroidPosX = Centroid().getCentroid(posX)
# negIndices = Centroid().getNFarthestPoints(unlabX, centroidPosX, len(posX))
#
# negX = []
# for i in negIndices:
#      negX.append(unlabX[i])

# newnewFeatureDict = {}
# for k, v in newLabelDict.iteritems():
#     if v == 1:
#         newnewFeatureDict[k] = newFeatureDict[k]
# for i in range(0, len(negX)):
#     newnewFeatureDict[str(i)] = negX[i]
#     newLabelDict[str(i)] = 0

# dataRetriever.convertToCSV(newFeatureDict, newLabelDict, "out2.csv")


crossVal = []
delKeys = []

count = 0
total = 0
for k, v in labelDict.iteritems():
    if v == 1:
        count += 1
        total += 1
    if count == 4:
        count = 0
        crossVal.append(featureVectorDict[k])
        delKeys.append(k)
        del featureVectorDict[k]
for key in delKeys:
    del labelDict[key]

x = PEBL(verbose=True)

# print("no gamma")
# x.train(featureVectorDict, labelDict)
# x.score(crossVal + unlabX, [1] * len(crossVal) + [0] * len(unlabX))
#
# print(".8 gamma")
# x.train(featureVectorDict, labelDict, gamma=.8)
# x.score(crossVal + unlabX, [1] * len(crossVal) + [0] * len(unlabX))
#
print("number of hidden positives: {0}".format(len(crossVal)))
print("Data size: {0} x {1}".format(len(featureVectorDict.values()[0]), len(featureVectorDict)))
x.train(featureVectorDict, labelDict, kernel="poly", probability=True, class_weight="auto", gamma=.2)
x.score(crossVal + unlabX, [1] * len(crossVal) + [0] * len(unlabX))
#
# print("linear")
# x.train(featureVectorDict, labelDict, kernel="linear")
# x.score(crossVal + unlabX, [1] * len(crossVal) + [0] * len(unlabX))
