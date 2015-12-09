from dataGrabber import dataGrabber, dataFileDescriptor
from Classifier import Classifier
from helpers import helpers
from logger import logger
from ExampleSampler import ExampleSampler
from featureSelection import featureSelection
import numpy

matchFeaturesFileName = "/home/alex/KnowEng/data/RIPPER_RULES_GO40FILT2000_KEGGALL_x10POS.txt"
matchFeatures = open(matchFeaturesFileName, 'r').read().split()
filterByMatchFeatures = lambda feature, count: feature in matchFeatures
filterTermsByCount = lambda feature, count: count >= 40 and count <= 2000

labelFile = "/home/alex/KnowEng/data/VANTVEER_BREAST_CANCER_ESR1.CB.txt"
featuresDataFile1 = dataFileDescriptor("/home/alex/KnowEng/data/ENSG.kegg_pathway.txt", filterByMatchFeatures)
featuresDataFile2 = dataFileDescriptor("/home/alex/KnowEng/data/ENSG.go_%_evid.txt", filterByMatchFeatures)
featuresDataFiles = [featuresDataFile1, featuresDataFile2]

logOutputDir = "/home/alex/KnowEng/logs/now/"
loggerObj = logger(baseDir=logOutputDir)
helperObj = helpers(loggerObj)

dataRetriever = dataGrabber(loggerObj)
(dict_X, dict_y, positiveKeys, negativeKeys, dataIndices) = dataRetriever.getData(labelFile, featuresDataFiles)

# sampler = ExampleSampler(dict_X, dict_y, loggerObj)
# featureSelector = featureSelection(loggerObj)
#
# featureIndices = featureSelector.rankFeaturesFourier(dict_X, dict_y, 1)
#
# removeIndices = list(set(range(0, len(featureIndices))) - set(featureIndices[0:360]))
# for k, v in dict_X.iteritems():
#     dict_X[k] = numpy.delete(numpy.array(v), removeIndices, 0).tolist()
#
# dict_X, dict_y, excludeKeys = sampler.smote(list(positiveKeys), 300, 5, 1)
#sampler.randomUnderSample(negativeKeys, .50)
excludeKeys = set()
x = Classifier(loggerObj, dict_X, dict_y, kCrossValPos=3, kCrossValNeg=0, crossValExcludeSet=excludeKeys, verbose=True)

for i in range(0, 11):
    x.trainSVM(kernel="poly", class_weight ='auto', gamma=i*.1, degree=3)
    x.score()
