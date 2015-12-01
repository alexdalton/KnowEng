from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
from fourier import Fourier
import numpy as np
import operator


class featureSelection():
    def __init__(self, logger):
        self.logger = logger

    def rankFeaturesChi2(self, X, y):
        return np.argsort(SelectKBest(chi2).fit(X, y).scores_)[::-1]

    def rankFeaturesForest(self, X, y, n_estimators):
        return np.argsort(ExtraTreesClassifier(n_estimators=n_estimators).fit(X, y).feature_importances_)[::-1]

    def rankFeaturesFourier(self, X, y, d=1):
        compare = lambda x,y: cmp(abs(x), abs(y))
        sorted_coeffs = sorted(Fourier(X, y, self.logger).coeff(d).items(), key=operator.itemgetter(1), reverse=True, cmp=compare)
        return sorted_coeffs