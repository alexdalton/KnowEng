import itertools
from math import factorial
import json
import operator
import matplotlib.pyplot as plt


class Fourier:

    def __init__(self, dict_X, dict_y, logger, verbose=False):
        self.logger = logger
        self.verbose = verbose
        self.featureDim = len(dict_X.values()[0])
        self.dict_X = dict_X
        self.dict_y = dict_y

    def getFourierFeatures(self, d=1, N=10, thresh=None, coeffsFileName=None):
        if coeffsFileName is None:
            compare = lambda x,y: cmp(abs(x), abs(y))
            sorted_coeffs = sorted(self.coeff(d).items(), key=operator.itemgetter(1), reverse=True, cmp=compare)
            json.dump(sorted_coeffs, open("fourier_{0}".format(d), "w"))
        else:
            sorted_coeffs = json.load(open(coeffsFileName))

        hist = []
        for subset_coeff in sorted_coeffs:
            hist.append(subset_coeff[1])

        # n, bins, patches = plt.hist(hist, 100)
        # plt.title('Fourier Coeffecient Distribution for d = {0}'.format(d))
        # plt.xlabel('Fourier Coeffecient Value')
        # plt.ylabel('Bin Count')
        # plt.show()
        # exit(0)

        new_dict_X = {}
        for key in self.dict_X.iterkeys():
            new_dict_X[key] = []

        if thresh:
            for subset_coeff in sorted_coeffs:
                if abs(subset_coeff[1]) < thresh:
                    break
                for key, featureVector in self.dict_X.iteritems():
                    new_dict_X[key].append(self.basis(self.featureSubset(featureVector, subset_coeff[0]), negReturn=0))
        else:
            for subset_coeff in sorted_coeffs[0:N]:
                for key, featureVector in self.dict_X.iteritems():
                    new_dict_X[key].append(self.basis(self.featureSubset(featureVector, subset_coeff[0]), negReturn=0))

        return new_dict_X

    def basis(self, boolArray, posReturn=1, negReturn=-1):
        if sum(boolArray) % 2 == 0:
            return posReturn
        else:
            return negReturn

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
            total += self.nCr(self.featureDim, i)
        return total

    def coeff(self, d):
        featureRange = range(0, self.featureDim)
        coeffs = {}

        totalCoeffs = self.totalCoeffs(d)

        count = 0
        for i in range(1, d + 1):
            subsets = set(itertools.combinations(featureRange, i))
            for subset in subsets:
                exampleKeys = self.dict_X.keys()
                a = 0
                for exampleKey in exampleKeys:
                    if self.dict_y[exampleKey] < 1:
                        label = -1
                    else:
                        label = 1
                    a += label * self.basis(self.featureSubset(self.dict_X[exampleKey], subset))
                coeffs[subset] = float(a) / float(len(exampleKeys))
                count += 1

                if self.verbose and count % 1000 == 0:
                    self.logger.log("{0:.2f}% Complete".format(100 * float(count) / float(totalCoeffs)))

        return coeffs
