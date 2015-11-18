from dataGrabber import dataGrabber, dataFileDescriptor
from helpers import helpers
from logger import logger

loggerObj = logger(shouldLog=False)
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

dataRetriever.convertToCSV(featureVectorDict, labelDict, "test.csv")