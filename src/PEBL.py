from sklearn import svm
from centroid import Centroid
from helpers import helpers
import time


class PEBL:

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.trainedSVM = None

    def score(self, X, y):
        if self.trainedSVM == None:
            print("Need to train PEBL first")
            return
        
        print("Predicting started")
        start = time.mktime(time.localtime())

        predictions = self.trainedSVM.predict(X)

        numPosEx = y.count(1)
        numCorrectPos = 0
        numPosPredictions = 0
        numCorrect = 0

        for i in range(0, len(predictions)):
            if predictions[i] == 1:
                numPosPredictions += 1
                if y[i] == 1:
                    numCorrectPos += 1
            if predictions[i] == y[i]:
                numCorrect += 1

        accuracy = float(numCorrect) / float(len(predictions))
        precision = float(numCorrectPos) / max(float(numPosPredictions), .1)
        recall = float(numCorrectPos) / max(float(numPosEx), .1)
        if recall > 0 and precision > 0:
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0
        print("Accuracy: {0}, Precision: {1}, Recall: {2}, F1: {3}".format(accuracy, precision, recall, f1))
        end = time.mktime(time.localtime())
        print("Predicting ended. Elapsed Time: {0} minutes".format((end - start) / 60))


    def train(self, featureDict, labelDict, kernel="rbf", gamma=0.0, probability=False, class_weight="auto"):
        """
        Trains an SVM using the PEBL algorithm
        """
        print("Training started")
        start = time.mktime(time.localtime())

        (positiveSet, unlabeledSet) = helpers().getLabelSets(labelDict)

        newNegativeFeatures = self._strongNegatives(list(unlabeledSet), positiveSet, featureDict)
        negativeSet = set()
        iterSVM = None

        if self.verbose:
            print("Number of positive examples: {0}".format(len(positiveSet)))
            print("Number of unlabeled examples: {0}".format(len(unlabeledSet)))

        iterCount = 0
        while(len(newNegativeFeatures)):
            iterCount += 1
            negativeSet = negativeSet | newNegativeFeatures
            iterSVM = svm.SVC(kernel=kernel, probability=probability, class_weight=class_weight, gamma=gamma)
            labels = ([1] * len(positiveSet)) + ([0] * len(negativeSet))
            featureVectors = helpers().dictOfFeaturesToList(featureDict, positiveSet) + \
                             helpers().dictOfFeaturesToList(featureDict, negativeSet)
            iterSVM.fit(featureVectors, labels)
            iterP = list(unlabeledSet - negativeSet)
            if len(iterP) == 0:
                break
            predictedLabels = iterSVM.predict(helpers().dictOfFeaturesToList(featureDict, iterP))
            newNegativeFeatures = set()
            for i in range(0, len(predictedLabels)):
                if predictedLabels[i] == 0:
                    newNegativeFeatures.add(iterP[i])
            if self.verbose:
                print("Iteration: {0}".format(iterCount))
                print("    negativeSet size: {0}".format(len(negativeSet)))
                print("    iterP size: {0}".format(len(iterP)))
                print("    newNegatives size: {0}".format(len(newNegativeFeatures)))

        self.trainedSVM = iterSVM
        end = time.mktime(time.localtime())
        print("Training Ended. Elapsed time: {0} minutes".format((end - start) / 60))
        return iterSVM

    def _strongNegatives(self, unlabeledSet, positiveSet, featureDict):
        posX = helpers().dictOfFeaturesToList(featureDict, positiveSet)
        unlabX = helpers().dictOfFeaturesToList(featureDict, unlabeledSet)

        centroidPosX = Centroid().getCentroid(posX)
        indices = Centroid().getNFarthestPoints(unlabX, centroidPosX, len(posX))

        strongNegatives = set()
        for i in indices:
            strongNegatives.add(unlabeledSet[i])

        return strongNegatives
