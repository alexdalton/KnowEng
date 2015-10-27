from sklearn import svm
from centroid import Centroid
from helpers import helpers


class PEBL:

    def __init__(self):
        pass

    def train(self, featureDict, labelDict):
        """
        Trains an SVM using the PEBL algorithm
        """
        (positiveSet, unlabeledSet) = helpers().getLabelSets(labelDict)

        newNegativeFeatures = self._strongNegatives(list(unlabeledSet), positiveSet, featureDict)
        negativeSet = set()
        iterSVM = None

        while(len(newNegativeFeatures)):
            negativeSet = negativeSet | newNegativeFeatures
            print len(negativeSet)
            iterSVM = svm.SVC(kernel="rbf", class_weight="auto")
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

        return iterSVM

    def _strongNegatives(self, unlabeledSet, positiveSet, featureDict):
        posX = helpers.dictOfFeaturesToList(featureDict, positiveSet)
        unlabX = helpers.dictOfFeaturesToList(featureDict, unlabeledSet)

        centroidPosX = Centroid().getCentroid(posX)
        indices = Centroid().getNFarthestPoints(unlabX, centroidPosX, len(posX))

        strongNegatives = set()
        for i in indices:
            strongNegatives.add(unlabeledSet[i])

        return strongNegatives
