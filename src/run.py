#!/usr/bin/env python

from dataGrabber import dataGrabber, dataFileDescriptor
from Classifier import Classifier
from helpers import helpers
from logger import logger
from ExampleSampler import ExampleSampler
from fourier import Fourier
import numpy
import random

matchFeaturesFileName = "/home/alex/KnowEng/data/RIPPER_RULES_GO40FILT2000_KEGGALL_x10POS.txt"
matchFeatures = open(matchFeaturesFileName, 'r').read().split()
filterByMatchFeatures = lambda feature, count: feature in matchFeatures
filterTermsByCount = lambda feature, count: count >= 40 and count <= 2000

labelFile = "/home/alex/KnowEng/data/geneSets/VANTVEER_BREAST_CANCER_ESR1.CB.txt"
featuresDataFile1 = dataFileDescriptor("/home/alex/KnowEng/data/edges/ENSG.kegg_pathway.txt", filterByMatchFeatures)
featuresDataFile2 = dataFileDescriptor("/home/alex/KnowEng/data/edges/ENSG.go_%_evid.txt", filterByMatchFeatures)
featuresDataFiles = [featuresDataFile1, featuresDataFile2]

logOutputDir = "/home/alex/KnowEng/logs/12-15/"
loggerObj = logger(baseDir=logOutputDir)
helperObj = helpers(loggerObj)

dataRetriever = dataGrabber(loggerObj)
(dict_X, dict_y, positiveKeys, negativeKeys, dataIndices) = dataRetriever.getData(labelFile, featuresDataFiles)

fourierFeatureSelector = Fourier(dict_X, dict_y, loggerObj)
dict_X = fourierFeatureSelector.getFourierFeatures(d=3, thresh=.93, coeffsFileName='/home/alex/KnowEng/data/fourier_3')

# sampler = ExampleSampler(dict_X, dict_y, loggerObj)
# dict_X, dict_y, excludeKeys = sampler.smote(list(positiveKeys), 200, 5, 1)

#sampler.randomUnderSample(negativeKeys, i * .15)
excludeKeys = set()

x = Classifier(loggerObj, dict_X, dict_y, kCrossValPos=3, kCrossValNeg=0, crossValExcludeSet=excludeKeys, verbose=True)

x.trainSVM(kernel="poly", class_weight ='auto', gamma=.4, degree=3)
x.score()
