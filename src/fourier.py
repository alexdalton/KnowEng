import random
import itertools
from math import log, ceil, factorial


class fourier:

    def __init__(self, featureDict, labelDict, featureDim):
        self.featureIds = featureDict.keys()
        self.featureDict = featureDict
        self.labelDict = labelDict
        self.featureDim = featureDim
        self.features = []
        self.labels = []
        for k, v in self.featureDict.iteritems():
            self.features.append(v)
            self.labels.append(self.labelDict[k])

    def _getLabelSets(self):
        positiveSet = set()
        negativeSet = set()
        for k, v in self.labelDict.iteritems():
            if v == 1:
                positiveSet.add(k)
            else:
                negativeSet.add(k)
        return (positiveSet, negativeSet)

    def calcM(self, lam, prob):
        return int(ceil(2 * log(prob / 2) / (-1 * lam * lam)))

    def getRandomFeatures(self, numFeatures):
        (positiveSet, negativeSet) = self._getLabelSets()
        randomIds = random.sample(negativeSet, numFeatures - len(positiveSet)) + list(positiveSet)
        #randomIds = random.sample(self.featureIds, numFeatures)
        features = []
        labels = []
        for randomId in randomIds:
            features.append(self.featureDict[randomId])
            labels.append(self.labelDict[randomId])
        return (features, labels)

    def basis(self, boolArray):
        x = 0
        for i in range(0, len(boolArray)):
            x = x + boolArray[i]
        if x % 2 == 0:
            return 1
        else:
            return -1

    def featureSubset(self, featureVector, subset):
        out = []
        for i in subset:
            out.append(featureVector[i])
        return out

    def nCr(self, n, r):
        return factorial(n) / factorial(r) / factorial(n-r)

    def totalCoeffs(self, d):
        total = 0
        for i in range(1, d + 1):
            total = total + self.nCr(self.featureDim, i)
        return total

    def coeff(self, d, lam, prob):
        featureRange = range(0, self.featureDim)
        m = self.calcM(lam, prob)
        coeffs = {}

        totalCoeffs = self.totalCoeffs(d)

        count = 0
        for i in range(1, d + 1):
            subsets = set(itertools.combinations(featureRange, i))
            for subset in subsets:
                #print subset
                #(features, labels) = self.getRandomFeatures(m)
                (features, labels) = (self.features, self.labels)
                a = 0
                for j in range(0, len(features)):
                    a = a + labels[j] * self.basis(self.featureSubset(features[j], subset))
                coeffs[subset] = abs(float(a) / float(len(features)))
                count = count + 1

                if count % 1000 == 0:
                    print("{0:.2f}% Complete".format(100 * float(count) / float(totalCoeffs)))
                #print(subset, coeffs[subset])

        return coeffs