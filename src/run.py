from dataGrabber import dataGrabber, dataFileDescriptor
from Classifier import Classifier
from helpers import helpers
from logger import logger
from ExampleSampler import ExampleSampler
from featureSelection import featureSelection
import numpy

matchFeatures = []
filterTermsByCount = lambda feature, count: count >= 40 and count <= 2000
filterByMatchFeatures = lambda feature, count: feature in matchFeatures

breastCancerLabelFile = "/home/alex/KnowEng/data/VANTVEER_BREAST_CANCER_ESR1.CB.txt"
keggDataFile = dataFileDescriptor("/home/alex/KnowEng/data/ENSG.kegg_pathway.txt")
goDataFile = dataFileDescriptor("/home/alex/KnowEng/data/ENSG.go_%_evid.txt", filterTermsByCount)

logOutputDir = "/home/alex/KnowEng/logs/fourier/"

loggerObj = logger(baseDir=logOutputDir)
helperObj = helpers(loggerObj)

#ripperRuleFileName = "/home/alex/KnowEng/data/RIPPER_RULES_GO40FILT2000_KEGGALL_x5POSITIVE.txt"
#matchFeatures = helperObj.getFeaturesFromRipperRules(ripperRuleFileName, ["go", "kegg"])

dataRetriever = dataGrabber(loggerObj)
(dict_X, dict_y, dataIndices) = dataRetriever.getData(breastCancerLabelFile, [keggDataFile, goDataFile])

#excludeKeys = set()

sampler = ExampleSampler(dict_X, dict_y, loggerObj)
positiveKeys, negativeKeys = helperObj.getLabelSets(dict_y)
#sampler.randomUnderSample(negativeKeys, .75)
#sampler.tomekUnderSample(positiveKeys, negativeKeys)
featureSelector = featureSelection(loggerObj)
ranked = featureSelector.rankFeaturesFourier(dict_X, dict_y, 1)

featureIndices = []
for ranking in ranked:
    for index in ranking[0]:
        if index not in featureIndices:
            featureIndices.append(index)

removeIndices = list(set(range(0, len(featureIndices))) - set(featureIndices[0:120]))
for k, v in dict_X.iteritems():
    dict_X[k] = numpy.delete(numpy.array(v), removeIndices, 0).tolist()

dict_X, dict_y, excludeKeys = sampler.smote(list(positiveKeys), 400, 5, 1)

x = Classifier(loggerObj, verbose=True)

x.trainSVM(dict_X, dict_y, kCrossValPos=3, kCrossValNeg=0, crossValExcludeSet=excludeKeys, kernel="rbf",
             class_weight ='auto', gamma=3, degree=3)
x.score()

# x.trainPEBLSVM(dict_X, dict_y, kCrossValPos=3, kCrossValNeg=0, crossValExcludeSet=excludeKeys, kernel="rbf",
#             class_weight ='auto', gamma=3, degree=3)
# x.score()
