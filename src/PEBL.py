from sklearn import svm
from sklearn.neighbors.nearest_centroid import NearestCentroid
from random import sample

class PEBL:

    def __init__(self):
        pass

    def _getLabelSets(self, labelDict):
        positiveSet = set()
        unlabeledSet = set()
        for k, v in labelDict.iteritems():
            if v == 1:
                positiveSet.add(k)
            else:
                unlabeledSet.add(k)
        return (positiveSet, unlabeledSet)

    def _getFeatureVectors(self, featureIds, featureDict):
        featureVectors = []
        for featureId in featureIds:
            featureVectors.append(featureDict[featureId])
        return featureVectors

    def train(self, featureDict, labelDict):
        """
        Trains an SVM using the PEBL algorithm
        """

        (positiveSet, unlabeledSet) = self._getLabelSets(labelDict)

        newNegativeFeatures = self._strongNegatives(unlabeledSet, positiveSet)
        negativeSet = set()
        iterSVM = None

        while(len(newNegativeFeatures)):
            negativeSet = negativeSet | newNegativeFeatures
            print len(negativeSet)
            iterSVM = svm.SVC(kernel="rbf", class_weight="auto")
            labels = ([1] * len(positiveSet)) + ([0] * len(negativeSet))
            featureVectors = self._getFeatureVectors(positiveSet, featureDict) + self._getFeatureVectors(negativeSet, featureDict)
            iterSVM.fit(featureVectors, labels)
            iterP = list(unlabeledSet - negativeSet)
            if len(iterP) == 0:
                break
            predictedLabels = iterSVM.predict(self._getFeatureVectors(iterP, featureDict))
            newNegativeFeatures = set()
            for i in range(0, len(predictedLabels)):
                if predictedLabels[i] == 0:
                    newNegativeFeatures.add(iterP[i])

        return iterSVM

    def _strongNegatives(self, unlabeledSet, positiveSet):
        """
        Should return the strongly negatives samples from the unlabeledSet
        Currently just returns a set of random samples from the unlabeledSet that's 1% in size
        :param unlabeledSet: set of unlabeled samples
        :param positiveSet: set of postive samples
        :return: set of strongly negative samples
        """
        return set(sample(unlabeledSet, int(.10 * len(unlabeledSet))))