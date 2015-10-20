from dataGrabber import dataGrabber
from PEBL import PEBL

labelFile = "/home/alex/KnowEng/data/LEE_LIVER_CANCER_ACOX1.CB.txt"
dataFile = "/home/alex/KnowEng/data/ENSG.kegg_pathway.txt"

dataRetriever = dataGrabber()
(featureVectorDict, labelDict) = dataRetriever.getData(labelFile, dataFile, 1, 0)

classifierPEBL = PEBL()
classifierPEBL.train(featureVectorDict, labelDict)

#labels = classifierPEBL.classify(positiveData + unlabeledData)