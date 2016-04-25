#!/usr/bin/env python

from optparse import OptionParser
import os
import ConfigParser
from dataGrabber import dataGrabber, dataFileDescriptor
from Classifier import Classifier
from helpers import helpers
from logger import logger
from datetime import date
import random

knowEngRoot = "/workspace/"


# Returns the option value from a given section within the configuration, else returns a default value
# Includes an option to split the value by comma seperator
def grabOptionOrDefault(config, section, option, default=None, split=False):
    if config.has_option(section, option):
        if split:
            return config.get(section, option).replace(" ", "").split(",")
        else:
            return config.get(section, option)
    else:
        return default

# Returns the option value from a given section within the configuration, else raises an error
# Includes an option to split the value by comma seperator
def grabOptionOrError(config, section, option, split=False):
    try:
        if split:
            return config.get(section, option).replace(" ", "").split(",")
        else:
            return config.get(section, option)
    except ConfigParser.NoOptionError:
        print("Missing {0} option for section {1}".format(option, section))
        return None

usage = "usage: %prog -f config_file_name"
parser = OptionParser(usage=usage)
parser.add_option("-f", "--file", dest="configFilename",
                  help="config file containing tests to be ran", metavar="FILE")

(options, args) = parser.parse_args()

if options.configFilename is None:
    parser.print_help()

configFileBaseDir = knowEngRoot + "KnowEng/configs"
configFilename = os.path.join(configFileBaseDir, options.configFilename)

if not os.path.exists(configFilename):
    print("No config file at {0}".format(configFilename))
    exit(0)

config = ConfigParser.RawConfigParser()
config.read(configFilename)

if not config.has_section("global"):
    print("No global configuration specified in config file")
    exit(0)

resultFileBaseDir = knowEngRoot + "KnowEng/results"
resultsFilename = os.path.join(resultFileBaseDir, config.get("global", "resultsFile"))
shouldAppend = config.get("global", "append") in ["True", "true", "1"]
resultsHeaders = ["Gene Set", "Feature Info", "Algorithm", "Algorithm Info", "Data Size", "Num pos",
                  "Num hidden pos", "Num neg", "Num hidden neg", "Hidden Pos Precision", "Hidden Pos Recall", "Hidden Pos F1",
                  "Hidden Neg Precision", "Hidden Neg Recall", "Hidden Neg F1", "Hidden True Pos",
                  "Hidden False Neg", "Hidden False Pos", "Hidden True Neg", "Hidden Accuracy",
                  "Train Pos Precision", "Train Pos Recall", "Train Pos F1", "Train Neg Precision",
                  "Train Neg Recall", "Train Neg F1", "Train True Pos", "Train False Neg",
                  "Train False Pos", "Train True Neg", "Train Accuracy"]

if shouldAppend:
    results_fp = open(resultsFilename, 'a')
else:
    results_fp = open(resultsFilename, 'w')
    results_fp.write(str(resultsHeaders).strip("[]").replace("'", "") + '\n')

for testName in config.sections():
    if testName == "global":
        continue

    # Get files needed to create labeled feature vectors
    labelFiles = [grabOptionOrError(config, testName, "labelFile")]
    if labelFiles[0] == "All":
        labelFiles = os.listdir(knowEngRoot + "KnowEng/data/geneSets/")

    for labelFile in labelFiles:
        output = []
        output.append(labelFile)
        if not labelFile == "random":
            labelFile = os.path.join(knowEngRoot + "KnowEng/data/geneSets/", labelFile)
        featureFiles = grabOptionOrError(config, testName, "featureFiles", split=True)
        featureFilters = grabOptionOrDefault(config, testName, "featureFilters", split=True, default=[])
        output.append(str(featureFiles).replace(",", ";") + '; ' + str(featureFilters).replace(",", ";"))

        randomLabelFile = None
        randomSet = None
        randomSetLabels = None
        if grabOptionOrDefault(config, testName, "score") == "random":
            while True:
                randomLabelFile = random.sample(os.listdir(knowEngRoot + "KnowEng/data/geneSets/"), 1)[0]
                if not randomLabelFile == labelFile:
                    randomLabelFile = os.path.join(knowEngRoot + "KnowEng/data/geneSets/", randomLabelFile)
                    randomSet = open(randomLabelFile, "r").read().split()
                    randomSetLabels = [1] * len(randomSet)
                    break

        # Get algorithm to run for this test
        algorithm = grabOptionOrError(config, testName, "algorithm")
        output.append(algorithm)

        # Get SVM variables
        kernel = str(grabOptionOrDefault(config, testName, "kernel", default="rbf"))
        gamma = float(grabOptionOrDefault(config, testName, "gamma", default=0.0))
        degree = int(grabOptionOrDefault(config, testName, "degree", default=3))
        class_weight = str(grabOptionOrDefault(config, testName, "class_weight", default="auto"))
        kCrossValPos = int(grabOptionOrDefault(config, testName, "kCrossValPos", default=3))
        kCrossValNeg = int(grabOptionOrDefault(config, testName, "kCrossValNeg", default=0))
        C = float(grabOptionOrDefault(config, testName, "C", default=1.0))
        probability = bool(grabOptionOrDefault(config, testName, "probability", default="False") in ["True", "true", "1"])
        shrinking = bool(grabOptionOrDefault(config, testName, "shrinking", default="True") in ["True", "true", "1"])
        coef0 = float(grabOptionOrDefault(config, testName, "coef0", default=0.0))

        # Get feature modifications
        shouldSMOTE = bool(grabOptionOrDefault(config, testName, "SMOTE", default="False") in ["True", "true", "1"])
        smote_N = int(grabOptionOrDefault(config, testName, "smote_N", default=100))
        smote_k = int(grabOptionOrDefault(config, testName, "smote_k", default=5))

        shouldUndersample = bool(grabOptionOrDefault(config, testName, "undersample", default="False") in ["True", "true", "1"])
        undersample_percent = float(grabOptionOrDefault(config, testName, "undersample_percent", default=.50))

        output.append("kernel={0}; gamma={1}; degree={2}; class_weight={3}; kCrossValPos={4}; kCrossValNeg={5}; "
                      "C={6}; probability={7}; shrinking={8}; coef0={9}; SMOTE={10}; N={11}; k={12}; "
                      "undersample={13}; percent={14}".format(kernel, gamma, degree, class_weight, kCrossValPos,
                                                            kCrossValNeg, C, probability, shrinking, coef0, shouldSMOTE,
                                                            smote_N, smote_k, shouldUndersample, undersample_percent))
        featureDataFiles = []
        featureFileBaseDir = knowEngRoot + "KnowEng/data/featureVectors/DCAdata"
        for i in range(0, len(featureFiles)):
            if len(featureFilters) > 0:
                if featureFilters[i] == "bySelection":
                    featureSelectionFile = str(grabOptionOrError(config, testName, "selectedFeaturesFile"))
                    filter = lambda feature, count: feature in open(os.path.join(knowEngRoot + "KnowEng/data/", featureSelectionFile), 'r').read().split()
                elif featureFilters[i] == "byCount":
                    filter = lambda feature, count: count >= 40 and count <= 2000
                else:
                    filter = lambda feature, count: True
                featureDataFiles.append(dataFileDescriptor(os.path.join(featureFileBaseDir, featureFiles[i]), filter))

        logOutputDir = knowEngRoot + "KnowEng/logs/" + date.today().isoformat()
        if not os.path.isdir(logOutputDir):
            os.mkdir(logOutputDir)

        loggerObj = logger(baseDir=logOutputDir)
        helperObj = helpers(loggerObj)
        dataRetriever = dataGrabber(loggerObj)
        featureFiles[0] = os.path.join(featureFileBaseDir, featureFiles[0])
        featureFiles[1] = os.path.join(featureFileBaseDir, featureFiles[1])
        (dict_X, dict_y, positiveKeys, negativeKeys) = dataRetriever.getDCAData(featureFiles[0], featureFiles[1], labelFile)

        doLASSO = bool(grabOptionOrDefault(config, testName, "LASSO", default="True") in ["True", "true", "1"])
        alpha = float(grabOptionOrDefault(config, testName, "alpha", default=1.0) )

        x = Classifier(loggerObj, dict_X, dict_y, shouldSMOTE=shouldSMOTE, smote_N=smote_N, smote_k=smote_k,
                       doLASSO=doLASSO, alpha=alpha, kCrossValPos=kCrossValPos, kCrossValNeg=kCrossValNeg, verbose=True)

        if algorithm == "SVM":
            x.trainSVM(kernel, C, gamma, probability, shrinking, class_weight, degree, coef0)
        else:
            x.trainPEBLSVM(kernel, C, gamma, probability, shrinking, class_weight, degree, coef0)

        output += x.score(randomSetName=randomLabelFile,randomSet=randomSet, randomSetLabels=randomSetLabels)

        results_fp.write(str(output).replace("[", "").replace("]", "") + '\n')
