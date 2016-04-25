from sklearn import svm
from centroid import Centroid
from helpers import helpers
import time
from tabulate import tabulate
from random import sample
from ExampleSampler import ExampleSampler
from sklearn import linear_model
import numpy as np


class Classifier:

    def __init__(self, logger, dict_X, dict_y, shouldSMOTE=False, smote_N=100, smote_k=5,
                 doLASSO=False, alpha=1.0, kCrossValPos=0, kCrossValNeg=0, verbose=False):
        # class variables to initialize general classifier
        self.logger = logger
        self.verbose = verbose
        self.helperObj = helpers(logger)

        self.dict_X = dict_X.copy()
        self.dict_y = dict_y.copy()
        self.kCrossValPos = kCrossValPos
        self.kCrossValNeg = kCrossValNeg
        self.hidden_X = []
        self.hidden_y = []
        self.totalExamples = len(self.dict_X)
        self.featureDims = len(self.dict_X.values()[0])
        (self.allPosExampleKeys, self.allNegExampleKeys) = self.helperObj.getLabelSets(dict_y)
        self.numAllPosExamples = len(self.allPosExampleKeys)
        self.numAllNegExamples = len(self.allNegExampleKeys)
        self.crossValExcludeSet = set()
        self._hideSamples()

        if shouldSMOTE:
            sampler = ExampleSampler(dict_X, dict_y, self.logger)
            trainPosKeys = list(set(self.allPosExampleKeys) - set(self.hiddenPosExampleKeys))
            self.dict_X, self.dict_y, self.crossValExcludeSet = sampler.smote(trainPosKeys, smote_N, smote_k, 1)
            self.crossValExcludeSet = set(self.crossValExcludeSet)
            self.totalExamples = len(self.dict_X)
            self.featureDims = len(self.dict_X.values()[0])
            (self.allPosExampleKeys, self.allNegExampleKeys) = self.helperObj.getLabelSets(dict_y)
            self.numAllPosExamples = len(self.allPosExampleKeys)
            self.numAllNegExamples = len(self.allNegExampleKeys)

        if doLASSO:
            positiveExampleKeys = set(self.allPosExampleKeys) - set(self.hiddenPosExampleKeys)
            negativeExampleKeys = set(self.allNegExampleKeys) - set(self.hiddenNegExampleKeys)
            y = ([1] * len(positiveExampleKeys)) + ([0] * len(negativeExampleKeys))
            X = self.helperObj.dictOfFeaturesToList(self.dict_X, positiveExampleKeys) + \
                self.helperObj.dictOfFeaturesToList(self.dict_X, negativeExampleKeys)
            clf = linear_model.Lasso(alpha=alpha, selection="random")
            clf.fit(X, y)

            coefs = np.array(clf.coef_)
            zeroCoefIndices = np.where(coefs == 0)[0].tolist()
            for key in self.dict_X.iterkeys():
                self.dict_X[key] = np.delete(self.dict_X[key], zeroCoefIndices, 0).tolist()
            self.hidden_X = np.delete(self.hidden_X, zeroCoefIndices, 1).tolist()
            self.featureDims = len(self.dict_X.values()[0])

        self.logger.log("Data size: {0} x {1}".format(self.featureDims, self.totalExamples))
        self.logger.log("Total Number of Positive Examples: {0}".format(self.numAllPosExamples))
        self.logger.log("Number of Hidden Positive Examples: {0}".format(self.numHiddenPosExamples))
        self.logger.log("Total Number of Negative Examples: {0}".format(self.numAllNegExamples))
        self.logger.log("Number of Hidden Negative Examples: {0}".format(self.numHiddenNegExamples))

        # holds the trained classifier after training
        self.trainedClassifier = None

    def _hideSamples(self):
        self.hiddenPosExampleKeys = []
        self.hiddenNegExampleKeys = []
        self.numHiddenPosExamples = 0
        self.numHiddenNegExamples = 0
        if self.kCrossValPos > 1:
            validPosExampleKeys = set(self.allPosExampleKeys) - self.crossValExcludeSet
            self.hiddenPosExampleKeys = sample(validPosExampleKeys, len(validPosExampleKeys) / self.kCrossValPos)
            self.numHiddenPosExamples = len(self.hiddenPosExampleKeys)
            self.logger.log("{0}-fold cross-validation on positive examples".format(self.kCrossValPos))

        if self.kCrossValNeg > 1:
            validNegExampleKeys = set(self.allNegExampleKeys) - self.crossValExcludeSet
            self.hiddenNegExampleKeys = sample(validNegExampleKeys, len(validNegExampleKeys) / self.kCrossValNeg)
            self.numHiddenNegExamples = len(self.hiddenNegExampleKeys)
            self.logger.log("{0}-fold cross-validation on negative examples".format(self.kCrossValNeg))

        # self.logger.log("Hidden positive keys:")
        # self.logger.log(str(self.hiddenPosExampleKeys))
        #
        # self.logger.log("Train positive keys:")
        # self.logger.log(str(set(self.allPosExampleKeys) - self.crossValExcludeSet - set(self.hiddenPosExampleKeys)))
        #
        # self.logger.log("Hidden negative keys:")
        # self.logger.log(str(self.hiddenNegExampleKeys))

        for exampleKey in self.hiddenPosExampleKeys + self.hiddenNegExampleKeys:
            self.hidden_X.append(self.dict_X[exampleKey])
            self.hidden_y.append(self.dict_y[exampleKey])
            del self.dict_X[exampleKey]
            del self.dict_y[exampleKey]

    def score(self, randomSetName=None, randomSet=None, randomSetLabels=None):
        if self.trainedClassifier is None:
            self.logger.log("Need to train an SVM first")
            return

        scores = ["{0} x {1}".format(self.featureDims, self.totalExamples), self.numAllPosExamples,
                  self.numHiddenPosExamples, self.numAllNegExamples, self.numHiddenNegExamples]

        if self.kCrossValNeg > 1 or self.kCrossValPos > 1:
            start = time.mktime(time.localtime())
            self.logger.log("Running classification on hidden examples")
            classifications = list(self.trainedClassifier.predict(self.hidden_X))
            end = time.mktime(time.localtime())
            self.logger.log("Elapsed Prediction Time: {0} minutes".format((end - start) / 60))
            scores += self.reportScores(classifications, self.hidden_y)
        else:
            scores += ['N/A'] * 11

        self.logger.log("Running classification on training examples")
        trainingExamples = []
        trainingLabels = []
        for posExampleKey in set(self.allPosExampleKeys) - set(self.hiddenPosExampleKeys) - self.crossValExcludeSet:
            trainingExamples.append(self.dict_X[posExampleKey])
            trainingLabels.append(1)
        for negExampleKey in set(self.allNegExampleKeys) - set(self.hiddenNegExampleKeys) - self.crossValExcludeSet:
            trainingExamples.append(self.dict_X[negExampleKey])
            trainingLabels.append(0)
        start = time.mktime(time.localtime())
        classifications = list(self.trainedClassifier.predict(trainingExamples))
        end = time.mktime(time.localtime())
        self.logger.log("Elapsed Prediction Time: {0} minutes".format((end - start) / 60))
        scores += self.reportScores(classifications, trainingLabels)

        if randomSet:
            for i in range(0, len(self.hiddenPosExampleKeys)):
                self.dict_X[self.hiddenPosExampleKeys[i]] = self.hidden_X[i]
            for i in range(0, len(self.hiddenNegExampleKeys)):
                self.dict_X[self.hiddenNegExampleKeys[i]] = self.hidden_X[i + len(self.hiddenPosExampleKeys)]
            self.logger.log("Running classification on random gene set: {0}".format(randomSetName))
            randomExamples = self.helperObj.dictOfFeaturesToList(self.dict_X, randomSet)
            start = time.mktime(time.localtime())
            classifications = list(self.trainedClassifier.predict(randomExamples))
            end = time.mktime(time.localtime())
            self.logger.log("Elapsed Prediction Time: {0} minutes".format((end - start) / 60))
            scores += [randomSetName] + self.reportScores(classifications, randomSetLabels)

        return scores

    def reportScores(self, predictedLabels, trueLabels):
        numPosEx = float(trueLabels.count(1))
        numNegEx = float(trueLabels.count(0))
        numCorrectPos = 0.0
        numCorrectNeg = 0.0
        numPosPredictions = float(predictedLabels.count(1))
        numNegPredictions = float(predictedLabels.count(0))

        for i in range(0, len(predictedLabels)):
            if predictedLabels[i] == trueLabels[i] == 1:
                numCorrectPos += 1.0
            if predictedLabels[i] == trueLabels[i] == 0:
                numCorrectNeg += 1.0

        try:
            accuracy = (numCorrectPos + numCorrectNeg) / float(len(predictedLabels))
        except ZeroDivisionError:
            self.logger.log("Number of predictions = 0")
            accuracy = float('nan')

        try:
            pprecision = numCorrectPos / numPosPredictions
        except ZeroDivisionError:
            self.logger.log("No positive predictions made")
            pprecision = float('nan')

        try:
            precall = float(numCorrectPos) / numPosEx
        except ZeroDivisionError:
            self.logger.log("No positive examples")
            precall = float('nan')

        try:
            pf1 = (2 * pprecision * precall) / (pprecision + precall)
        except ZeroDivisionError:
            self.logger.log("PPrecision and precall = 0")
            pf1 = float('nan')

        try:
            nprecision = numCorrectNeg / numNegPredictions
        except ZeroDivisionError:
            self.logger.log("No negative predictions made")
            nprecision = float('nan')

        try:
            nrecall = float(numCorrectNeg) / numNegEx
        except ZeroDivisionError:
            self.logger.log("No positive examples")
            nrecall = float('nan')

        try:
            nf1 = (2 * nprecision * nrecall) / (nprecision + nrecall)
        except ZeroDivisionError:
            self.logger.log("Precision and recall = 0")
            nf1 = float('nan')

        headers = ["Class", "Precision", "Recall", "F1"]
        table = [["POSITIVE", pprecision, precall, pf1],
                 ["NEGATIVE", nprecision, nrecall, nf1]]
        self.logger.log('\n' + tabulate(table, headers=headers))

        headers = ["", "Predicted POSITIVE", "Predicted NEGATIVE"]
        numIncorrectPos = numNegPredictions - numCorrectNeg
        numIncorrectNeg = numPosPredictions - numCorrectPos
        table = [["Actual POSITIVE", numCorrectPos, numIncorrectPos],
                 ["Actual NEGATIVE", numIncorrectNeg, numCorrectNeg]]
        self.logger.log('\n' + tabulate(table, headers=headers))

        self.logger.log("\nOverall accuracy: {0}".format(accuracy))

        return [pprecision, precall, pf1, nprecision, nrecall, nf1, numCorrectPos,
                numIncorrectPos, numIncorrectNeg, numCorrectNeg, accuracy]

    def _printVerboseTrainInfo(self, C, kernel, degree, gamma, coef0, probability, shrinking, class_weight):
        if self.verbose:
            degree_str = str(degree)
            coef0_str = str(coef0)
            if gamma == 0.0:
                gamma_str = str(1.0 / self.featureDims)
            else:
                gamma_str = str(gamma)

            if kernel != "sigmoid" and kernel != "poly":
                coef0_str = "N\A"
            if kernel != "poly":
                degree_str = "N\A"
            if kernel != "rbf" and kernel != "poly" and kernel != "sigmoid":
                gamma_str = "N\A"

            headers = ["Kernel", "C", "Gamma", "Degree", "Coef0", "Probability", "Class Weight", "Shrinking"]
            table = [[str(kernel), str(C), gamma_str, degree_str, coef0_str, str(probability), str(class_weight),
                      str(shrinking)]]
            self.logger.log(tabulate(table, headers=headers) + '\n')


    def trainSVM(self, kernel="rbf", C=1.0, gamma=0.0, probability=False, shrinking=True, class_weight="auto",
                 degree=3, coef0=0.0):
        self.logger.log("Algorithm: SVM\n")
        self._printVerboseTrainInfo(C, kernel, degree, gamma, coef0, probability, shrinking, class_weight)

        start = time.mktime(time.localtime())

        positiveExampleKeys = set(self.allPosExampleKeys) - set(self.hiddenPosExampleKeys)
        negativeExampleKeys = set(self.allNegExampleKeys) - set(self.hiddenNegExampleKeys)

        y = ([1] * len(positiveExampleKeys)) + ([0] * len(negativeExampleKeys))

        X = self.helperObj.dictOfFeaturesToList(self.dict_X, positiveExampleKeys) + \
            self.helperObj.dictOfFeaturesToList(self.dict_X, negativeExampleKeys)

        svc = svm.SVC(C=C, kernel=kernel, probability=probability, shrinking=shrinking,
                      class_weight=class_weight, gamma=gamma, degree=degree, coef0=coef0)

        svc.fit(X, y)

        self.trainedClassifier = svc
        end = time.mktime(time.localtime())
        self.logger.log("Elapsed training time: {0} minutes".format((end - start) / 60))
        return svc

    def trainPEBLSVM(self, kernel="rbf", C=1.0, gamma=0.0, probability=False, shrinking=True, class_weight="auto",
                     degree=3, coef0=0.0):
        self.logger.log("Algorithm: PEBL\n")
        self._printVerboseTrainInfo(C, kernel, degree, gamma, coef0, probability, shrinking, class_weight)

        start = time.mktime(time.localtime())

        positiveExampleKeys = set(self.allPosExampleKeys) - set(self.hiddenPosExampleKeys)
        unlabeledExampleKeys = set(self.allNegExampleKeys) - set(self.hiddenNegExampleKeys)

        newNegativeFeatures = self._strongNegatives(list(unlabeledExampleKeys), positiveExampleKeys)
        negativeSet = set()
        iterSVM = None

        iterCount = 0
        while len(newNegativeFeatures):
            iterCount += 1
            negativeSet = negativeSet | newNegativeFeatures
            iterSVM = svm.SVC(kernel=kernel, probability=probability,
                              class_weight=class_weight, gamma=gamma, degree=degree, coef0=coef0)
            labels = ([1] * len(positiveExampleKeys)) + ([0] * len(negativeSet))
            featureVectors = self.helperObj.dictOfFeaturesToList(self.dict_X, positiveExampleKeys) + \
                             self.helperObj.dictOfFeaturesToList(self.dict_X, negativeSet)
            iterSVM.fit(featureVectors, labels)
            iterP = list(unlabeledExampleKeys - negativeSet)
            if len(iterP) == 0:
                break
            predictedLabels = iterSVM.predict(self.helperObj.dictOfFeaturesToList(self.dict_X, iterP))
            newNegativeFeatures = set()
            for i in range(0, len(predictedLabels)):
                if predictedLabels[i] == 0:
                    newNegativeFeatures.add(iterP[i])
            if self.verbose:
                self.logger.log("Iteration: {0}".format(iterCount))
                self.logger.log("    negativeSet size: {0}".format(len(negativeSet)))
                self.logger.log("    iterP size: {0}".format(len(iterP)))
                self.logger.log("    newNegatives size: {0}".format(len(newNegativeFeatures)))

        self.trainedClassifier = iterSVM
        end = time.mktime(time.localtime())
        self.logger.log("Elapsed training time: {0} minutes".format((end - start) / 60))
        return iterSVM

    def _strongNegatives(self, unlabeledSet, positiveSet):
        posX = self.helperObj.dictOfFeaturesToList(self.dict_X, set(positiveSet) - self.crossValExcludeSet)
        unlabX = self.helperObj.dictOfFeaturesToList(self.dict_X, set(unlabeledSet) - self.crossValExcludeSet)

        centroidPosX = Centroid().getCentroid(posX)
        indices = Centroid().getNFarthestPoints(unlabX, centroidPosX, len(posX))

        strongNegatives = set()
        for i in indices:
            strongNegatives.add(unlabeledSet[i])

        return strongNegatives
