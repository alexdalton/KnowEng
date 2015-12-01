from sklearn import svm
from centroid import Centroid
from helpers import helpers
import time
from tabulate import tabulate
from random import sample


class Classifier:

    def __init__(self, logger, verbose=False):
        # class variables to initialize general classifier
        self.logger = logger
        self.verbose = verbose
        self.helperObj = helpers(logger)

        # class variables to initialize once a classificion model is specified
        # (i.e. initialized after a train method has been called)
        self.kCrossValPos = 1
        self.kCrossValNeg = 1
        self.totalExamples = 0
        self.featureDims = 0
        self.dict_X = {}
        self.dict_y = {}
        self.allPosExampleKeys = []
        self.allNegExampleKeys = []
        self.numAllPosExamples = 0
        self.numAllNegExamples = 0
        self.hidden_X = []
        self.hidden_y = []
        self.hiddenPosExampleKeys = []
        self.hiddenNegExampleKeys = []
        self.numHiddenPosExamples = 0
        self.numHiddenNegExamples = 0

        # holds the trained classifier after training
        self.trainedClassifier = None

    def _initNewClassificationModel(self, dict_X, dict_y, kCrossValPos, kCrossValNeg, crossValExcludeSet):
        self.dict_X = dict_X
        self.dict_y = dict_y
        self.kCrossValPos = kCrossValPos
        self.kCrossValNeg = kCrossValNeg
        self.totalExamples = len(dict_X)
        self.featureDims = len(dict_X.values()[0])
        (self.allPosExampleKeys, self.allNegExampleKeys) = self.helperObj.getLabelSets(dict_y)
        self.numAllPosExamples = len(self.allPosExampleKeys)
        self.numAllNegExamples = len(self.allNegExampleKeys)
        self.crossValExcludeSet = set(crossValExcludeSet)
        self._hideSamples(crossValExcludeSet)

    def _hideSamples(self, crossValExcludeSet):
        if self.kCrossValPos > 1:
            newPosExampleKeys = set(self.allPosExampleKeys) - set(crossValExcludeSet)
            self.hiddenPosExampleKeys = sample(newPosExampleKeys, len(newPosExampleKeys) / self.kCrossValPos)
            self.numHiddenPosExamples = len(self.hiddenPosExampleKeys)
            self.logger.log("{0}-fold cross-validation on positive examples".format(self.kCrossValPos))

        if self.kCrossValNeg > 1:
            newNegExampleKeys = set(self.allNegExampleKeys) - set(crossValExcludeSet)
            self.hiddenNegExampleKeys = sample(newNegExampleKeys, len(newNegExampleKeys) / self.kCrossValNeg)
            self.numHiddenNegExamples = len(self.hiddenNegExampleKeys)
            self.logger.log("{0}-fold cross-validation on negative examples".format(self.kCrossValNeg))

        for exampleKey in self.hiddenPosExampleKeys + self.hiddenNegExampleKeys:
            self.hidden_X.append(self.dict_X[exampleKey])
            self.hidden_y.append(self.dict_y[exampleKey])
            del self.dict_X[exampleKey]
            del self.dict_y[exampleKey]

        self.logger.log("")

    def score(self):
        if self.trainedClassifier == None:
            self.logger.log("Need to train an SVM first")
            return

        start = time.mktime(time.localtime())

        predPosStr = "the hidden positive examples"
        # If no cross validation on positive examples, add all positives to hidden X and y
        if self.kCrossValPos <= 1:
            predPosStr = "all the given positive training examples"
            for posExampleKey in set(self.allPosExampleKeys) - self.crossValExcludeSet:
                self.hidden_X.append(self.dict_X[posExampleKey])
                self.hidden_y.append(1)

        predNegStr = "the hidden negative examples"
        # If no cross validation on negative examples, add all negatives to hidden X and y
        if self.kCrossValNeg <= 1:
            predNegStr = "all the given negative training examples"
            for negExampleKey in set(self.allNegExampleKeys) - self.crossValExcludeSet:
                self.hidden_X.append(self.dict_X[negExampleKey])
                self.hidden_y.append(0)

        self.logger.log("Predicting on {0} and {1}".format(predPosStr, predNegStr))
        predictions = list(self.trainedClassifier.predict(self.hidden_X))

        end = time.mktime(time.localtime())
        self.logger.log("Elapsed Prediction Time: {0} minutes".format((end - start) / 60))

        numPosEx = float(self.hidden_y.count(1))
        numNegEx = float(self.hidden_y.count(0))
        numCorrectPos = 0.0
        numCorrectNeg = 0.0
        numPosPredictions = float(predictions.count(1))
        numNegPredictions = float(predictions.count(0))

        for i in range(0, len(predictions)):
            if predictions[i] == self.hidden_y[i] == 1:
                numCorrectPos += 1.0
            if predictions[i] == self.hidden_y[i] == 0:
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
        table = [["POSITIVE", pprecision, precall, pf1],
                 ["NEGATIVE", nprecision, nrecall, nf1]]
        self.logger.log('\n' + tabulate(table, headers=headers))

        headers = ["", "Predicted POSITIVE", "Predicted NEGATIVE"]
        table = [["Actual POSITIVE", numCorrectPos, numNegPredictions - numCorrectNeg],
                 ["Actual NEGATIVE", numPosPredictions - numCorrectPos, numCorrectNeg]]
        self.logger.log('\n' + tabulate(table, headers=headers))

        self.logger.log("\nOverall accuracy: {0}".format(accuracy))

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
            self.logger.log("Data size: {0} x {1}".format(self.featureDims, self.totalExamples))
            self.logger.log("Total Number of Positive Examples: {0}".format(self.numAllPosExamples))
            self.logger.log("Number of Hidden Positive Examples: {0}".format(self.numHiddenPosExamples))
            self.logger.log("Total Number of Negative Examples: {0}".format(self.numAllNegExamples))
            self.logger.log("Number of Hidden Negative Examples: {0}".format(self.numHiddenNegExamples))

    def trainSVM(self, dict_X, dict_y, kCrossValPos=0, kCrossValNeg=0, crossValExcludeSet=set(), C=1.0, kernel="rbf", gamma=0.0,
                 probability=False, shrinking=True, class_weight="auto", degree=3, coef0=0.0):
        self.logger.log("Algorithm: SVM\n")
        self._initNewClassificationModel(dict_X, dict_y, kCrossValPos, kCrossValNeg, crossValExcludeSet)
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

    def trainPEBLSVM(self, dict_X, dict_y, kCrossValPos=0, kCrossValNeg=0, crossValExcludeSet=set(), C=1.0, kernel="rbf", gamma=0.0,
                     probability=False, shrinking=True, class_weight="auto", degree=3, coef0=0.0):
        self.logger.log("Algorithm: PEBL\n")
        self._initNewClassificationModel(dict_X, dict_y, kCrossValPos, kCrossValNeg, crossValExcludeSet)
        self._printVerboseTrainInfo(C, kernel, degree, gamma, coef0, probability, shrinking, class_weight)

        start = time.mktime(time.localtime())

        positiveExampleKeys = set(self.allPosExampleKeys) - set(self.hiddenPosExampleKeys)
        unlabeledExampleKeys = set(self.allNegExampleKeys) - set(self.hiddenNegExampleKeys)

        newNegativeFeatures = self._strongNegatives(list(unlabeledExampleKeys), positiveExampleKeys)
        negativeSet = set()
        iterSVM = None

        iterCount = 0
        while(len(newNegativeFeatures)):
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
