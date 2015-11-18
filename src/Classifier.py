from sklearn import svm
from centroid import Centroid
from helpers import helpers
import time
from tabulate import tabulate


class Classifier:

    def __init__(self, logger, verbose=False):
        self.logger = logger
        self.verbose = verbose
        self.trainedSVM = None
        self.helperObj = helpers(logger)

    def score(self, X, y):
        if self.trainedSVM == None:
            self.logger.log("Need to train an SVM first")
            return
        
        start = time.mktime(time.localtime())

        predictions = list(self.trainedSVM.predict(X))

        end = time.mktime(time.localtime())
        self.logger.log("Elapsed Prediction Time: {0} minutes".format((end - start) / 60))

        numPosEx = float(y.count(1))
        numNegEx = float(y.count(0))
        numCorrectPos = 0.0
        numCorrectNeg = 0.0
        numPosPredictions = float(predictions.count(1))
        numNegPredictions = float(predictions.count(0))

        for i in range(0, len(predictions)):
            if predictions[i] == y[i] == 1:
                numCorrectPos += 1.0
            if predictions[i] == y[i] == 0:
                numCorrectNeg += 1.0

        try:
            accuracy = (numCorrectPos + numCorrectNeg) / float(len(predictions))
        except ZeroDivisionError:
            self.logger.log("Number of predictions = 0")
            accuracy = 0

        try:
            pprecision = numCorrectPos / numPosPredictions
        except ZeroDivisionError:
            self.logger.log("No positive predictions made")
            pprecision = 0

        try:
            precall = float(numCorrectPos) / numPosEx
        except ZeroDivisionError:
            self.logger.log("No positive examples")
            precall = 0

        try:
            pf1 = (2 * pprecision * precall) / (pprecision + precall)
        except ZeroDivisionError:
            self.logger.log("PPrecision and precall = 0")
            pf1 = 1

        try:
            nprecision = numCorrectNeg / numNegPredictions
        except ZeroDivisionError:
            self.logger.log("No negative predictions made")
            nprecision = 0

        try:
            nrecall = float(numCorrectNeg) / numNegEx
        except ZeroDivisionError:
            self.logger.log("No positive examples")
            nrecall = 0

        try:
            nf1 = (2 * nprecision * nrecall) / (nprecision + nrecall)
        except ZeroDivisionError:
            self.logger.log("Precision and recall = 0")
            nf1 = 1

        headers = ["Class", "Precision", "Recall", "F1"]
        table = [["+", pprecision, precall, pf1],
                 ["-", nprecision, nrecall, nf1]]
        self.logger.log('\n' + tabulate(table, headers=headers))

        headers = ["", "Predicted +", "Predicted -"]
        table = [["Actual +", numCorrectPos, numNegPredictions - numCorrectNeg],
                 ["Actual -", numPosPredictions - numCorrectPos, numCorrectNeg]]
        self.logger.log('\n' + tabulate(table, headers=headers))

        self.logger.log("\nOverall accuracy: {0}".format(accuracy))

    def printVerboseTrainInfo(self, featureDict, labelDict, positiveSet, unlabeledSet, kernel, gamma,
                              probability, shrinking, class_weight, degree, coef0):
        if self.verbose:
            headers = ["Kernel", "Gamma", "Probability", "Class Weight", "Shrinking", "Degree", "Coef0"]
            degree_str = str(degree)
            if gamma == 0.0:
                gamma_str = str(1.0 / len(featureDict.values()[0]))
            else:
                gamma_str = str(gamma)
            coef0_str = str(coef0)
            if kernel != "sigmoid" and kernel != "poly":
                coef0_str = "N\A"
            if kernel != "poly":
                degree_str = "N\A"
            if kernel != "rbf" and kernel != "poly" and kernel != "sigmoid":
                gamma_str = "N\A"
            table = [[kernel, gamma_str, str(probability), class_weight, str(shrinking), degree_str, coef0_str]]
            self.logger.log(tabulate(table, headers=headers) + '\n')
            self.logger.log("Data size: {0} x {1}".format(len(featureDict.values()[0]), len(featureDict)))
            self.logger.log("Number of Positive Examples: {0}".format(len(positiveSet)))
            self.logger.log("Number of Negative/Unlabeled Examples: {0}".format(len(unlabeledSet)))

    def trainSVM(self, featureDict, labelDict, kernel="rbf", gamma=0.0,
                 probability=False, shrinking=True, class_weight="auto", degree=3, coef0=0.0):
        self.logger.log("Algorithm: SVM\n")
        (positiveSet, unlabeledSet) = self.helperObj.getLabelSets(labelDict)

        self.printVerboseTrainInfo(featureDict, labelDict, positiveSet, unlabeledSet, kernel, gamma, probability,
                                   shrinking, class_weight, degree, coef0)

        start = time.mktime(time.localtime())

        labels = ([1] * len(positiveSet)) + ([0] * len(unlabeledSet))

        featureVectors = self.helperObj.dictOfFeaturesToList(featureDict, positiveSet) + \
                         self.helperObj.dictOfFeaturesToList(featureDict, unlabeledSet)

        svc = svm.SVC(kernel=kernel, probability=probability, shrinking=shrinking,
                      class_weight=class_weight, gamma=gamma, degree=degree, coef0=coef0)

        svc.fit(featureVectors, labels)

        self.trainedSVM = svc
        end = time.mktime(time.localtime())
        self.logger.log("Elapsed training time: {0} minutes".format((end - start) / 60))
        return svc

    def trainPEBLSVM(self, featureDict, labelDict, kernel="rbf", gamma=0.0,
              probability=False, shrinking=True, class_weight="auto", degree=3, coef0=0.0):
        self.logger.log("Algorithm: PEBL\n")

        start = time.mktime(time.localtime())

        (positiveSet, unlabeledSet) = self.helperObj.getLabelSets(labelDict)

        newNegativeFeatures = self._strongNegatives(list(unlabeledSet), positiveSet, featureDict)
        negativeSet = set()
        iterSVM = None

        self.printVerboseTrainInfo(featureDict, labelDict, positiveSet, unlabeledSet, kernel, gamma, probability,
                                   shrinking, class_weight, degree, coef0)

        iterCount = 0
        while(len(newNegativeFeatures)):
            iterCount += 1
            negativeSet = negativeSet | newNegativeFeatures
            iterSVM = svm.SVC(kernel=kernel, probability=probability,
                              class_weight=class_weight, gamma=gamma, degree=degree, coef0=coef0)
            labels = ([1] * len(positiveSet)) + ([0] * len(negativeSet))
            featureVectors = self.helperObj.dictOfFeaturesToList(featureDict, positiveSet) + \
                             self.helperObj.dictOfFeaturesToList(featureDict, negativeSet)
            iterSVM.fit(featureVectors, labels)
            iterP = list(unlabeledSet - negativeSet)
            if len(iterP) == 0:
                break
            predictedLabels = iterSVM.predict(self.helperObj.dictOfFeaturesToList(featureDict, iterP))
            newNegativeFeatures = set()
            for i in range(0, len(predictedLabels)):
                if predictedLabels[i] == 0:
                    newNegativeFeatures.add(iterP[i])
            if self.verbose:
                self.logger.log("Iteration: {0}".format(iterCount))
                self.logger.log("    negativeSet size: {0}".format(len(negativeSet)))
                self.logger.log("    iterP size: {0}".format(len(iterP)))
                self.logger.log("    newNegatives size: {0}".format(len(newNegativeFeatures)))

        self.trainedSVM = iterSVM
        end = time.mktime(time.localtime())
        self.logger.log("Elapsed training time: {0} minutes".format((end - start) / 60))
        return iterSVM

    def _strongNegatives(self, unlabeledSet, positiveSet, featureDict):
        posX = self.helperObj.dictOfFeaturesToList(featureDict, positiveSet)
        unlabX = self.helperObj.dictOfFeaturesToList(featureDict, unlabeledSet)

        centroidPosX = Centroid().getCentroid(posX)
        indices = Centroid().getNFarthestPoints(unlabX, centroidPosX, len(posX))

        strongNegatives = set()
        for i in indices:
            strongNegatives.add(unlabeledSet[i])

        return strongNegatives
