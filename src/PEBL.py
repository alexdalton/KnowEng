from sklearn import svm


class PEBL:

    def __init__(self, positiveFeatures, unlabeledFeatures):
        self.positiveFeatures = positiveFeatures
        self.unlabeledFeatures = unlabeledFeatures
        self.trainedSVM = None

    def train(self):
        """
        Trains an SVM using the PEBL algorithm
        """
        newNegativeFeatures = self._strongNegatives()
        negativeFeatures = []
        iterSVM = None

        while(len(newNegativeFeatures) > 0):
            negativeFeatures = list(set(negativeFeatures).union(set(newNegativeFeatures)))
            iterSVM = svm.SVC()
            labels = ([1] * len(self.positiveFeatures)) + ([0] * len(negativeFeatures))
            iterSVM.fit(self.positiveFeatures + negativeFeatures, labels)
            iterP = list(set(self.unlabeledFeatures).difference(set(negativeFeatures)))
            predictedLabels = iterSVM.predict(iterP)
            newNegativeFeatures = []
            for i in range(0, len(predictedLabels)):
                if predictedLabels[i] == 0:
                    newNegativeFeatures.append(iterP[i])

        self.trainedSVM = iterSVM

    def classify(self, features):
        """
        Classifies a list of features using the trained SVM
        :param features: list of features to be classified
        :return: a list of predicted labels for the features
        """

        if self.trainedSVM == None:
            return []

        return self.trainedSVM.predict(features)


    def _strongNegatives(self):
        return self.unlabeledFeatures