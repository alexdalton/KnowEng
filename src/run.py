from dataGrabber import dataGrabber
from PEBL import PEBL

labelFile = "/home/alex/KnowEng/data/LEE_LIVER_CANCER_ACOX1.CB.txt"
dataFile = "/home/alex/KnowEng/data/ENSG.kegg_pathway.txt"

dataRetriever = dataGrabber()
(positiveData, unlabeledData) = dataRetriever.getData(labelFile, dataFile, 1, 0)

classifierPEBL = PEBL(positiveData, unlabeledData)
classifierPEBL.train()

labels = classifierPEBL.classify(positiveData + unlabeledData)