from dataGrabber import dataGrabber, dataFileDescriptor
from Classifier import Classifier
from helpers import helpers
from logger import logger

filterTermsByCount = lambda feature, count: count >= 40 and count <= 2000
filterByMatchFeatures = lambda feature, count: feature in matchFeatures

breastCancerLabelFile = "/home/alex/KnowEng/data/VANTVEER_BREAST_CANCER_ESR1.CB.txt"
keggDataFile = dataFileDescriptor("/home/alex/KnowEng/data/ENSG.kegg_pathway.txt", filterByMatchFeatures)
goDataFile = dataFileDescriptor("/home/alex/KnowEng/data/ENSG.go_%_evid.txt", filterByMatchFeatures)

logOutputDir = "/home/alex/KnowEng/logs/svm/"

loggerObj = logger(baseDir=logOutputDir)
helperObj = helpers(loggerObj)

ripperRuleFileName = "/home/alex/KnowEng/data/RIPPER_RULES_GO40FILT2000_KEGGALL_x5POSITIVE.txt"
matchFeatures = helperObj.getFeaturesFromRipperRules(ripperRuleFileName, ["go", "kegg"])

dataRetriever = dataGrabber(loggerObj)
(featureVectorDict, labelDict, dataIndices) = dataRetriever.getData(breastCancerLabelFile, [keggDataFile, goDataFile])

x = Classifier(loggerObj, verbose=True)

x.trainSVM(featureVectorDict, labelDict, kCrossValPos=0, kCrossValNeg=0, kernel="rbf",
           class_weight ='auto', gamma=0.0, degree=3)
x.score()
