from dataGrabber import dataGrabber, dataFileDescriptor
from helpers import helpers
from logger import logger

loggerObj = logger(shouldLog=False)
helperObj = helpers(loggerObj)

filterTermsByCount = lambda feature, count: count >= 40 and count <= 2000

breastCancerLabelFile = "/home/alex/KnowEng/data/VANTVEER_BREAST_CANCER_ESR1.CB.txt"
keggDataFile = dataFileDescriptor("/home/alex/KnowEng/data/ENSG.kegg_pathway.txt")
goDataFile = dataFileDescriptor("/home/alex/KnowEng/data/ENSG.go_%_evid.txt", filterTermsByCount)

dataRetriever = dataGrabber(loggerObj)
(featureVectorDict, labelDict, dataIndices) = dataRetriever.getData(breastCancerLabelFile, [goDataFile, keggDataFile])

dataRetriever.convertToCSV(featureVectorDict, labelDict, "test.csv", 45)