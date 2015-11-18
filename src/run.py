from dataGrabber import dataGrabber, dataFileDescriptor
from Classifier import Classifier
from featureSelection import featureSelection
from centroid import Centroid
from helpers import helpers
from logger import logger

loggerObj = logger()
helperObj = helpers(loggerObj)

ripperRuleFileName = "/home/alex/KnowEng/data/RIPPER_RULES_GO40FILT2000_KEGGALL_x5POSITIVE.txt"
matchFeatures = helperObj.getFeaturesFromRipperRules(ripperRuleFileName, ["go", "kegg"])

filterTermsByCount = lambda feature, count: count >= 40 and count <= 2000
filterByMatchFeatures = lambda feature, count: feature in matchFeatures

breastCancerLabelFile = "/home/alex/KnowEng/data/VANTVEER_BREAST_CANCER_ESR1.CB.txt"
keggDataFile = dataFileDescriptor("/home/alex/KnowEng/data/ENSG.kegg_pathway.txt",
                                  filterByMatchFeatures)
goDataFile = dataFileDescriptor("/home/alex/KnowEng/data/ENSG.go_%_evid.txt",
                                filterByMatchFeatures)


dataRetriever = dataGrabber(loggerObj)
(featureVectorDict, labelDict, dataIndices) = dataRetriever.getData(breastCancerLabelFile, [goDataFile, keggDataFile])


#dataRetriever.convertToCSV(featureVectorDict, labelDict, "test.csv")

posX = helperObj.getFeaturesByLabel(featureVectorDict, labelDict, 1)
unlabX = helperObj.getFeaturesByLabel(featureVectorDict, labelDict, 0)
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


# crossVal = []
# delKeys = []
#
# count = 0
# total = 0
# for k, v in labelDict.iteritems():
#     if v == 1:
#         count += 1
#         total += 1
#     if count == 4:
#         count = 0
#         crossVal.append(featureVectorDict[k])
#         delKeys.append(k)
#         del featureVectorDict[k]
# for key in delKeys:
#     del labelDict[key]

x = Classifier(loggerObj, verbose=True)

# print("no gamma")
# x.train(featureVectorDict, labelDict)
# x.score(crossVal + unlabX, [1] * len(crossVal) + [0] * len(unlabX))
#
# print(".8 gamma")
# x.train(featureVectorDict, labelDict, gamma=.8)
# x.score(crossVal + unlabX, [1] * len(crossVal) + [0] * len(unlabX))
#
#print("number of hidden positives: {0}".format(len(crossVal)))

x.trainSVM(featureVectorDict, labelDict, kernel="rbf", class_weight ='auto', gamma=0.0, degree=3)
x.score(posX + unlabX, [1] * len(posX) + [0] * len(unlabX))

#
# print("linear")
# x.train(featureVectorDict, labelDict, kernel="linear")
# x.score(crossVal + unlabX, [1] * len(crossVal) + [0] * len(unlabX))
