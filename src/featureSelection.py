from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
from fourier import Fourier
import numpy as np
import operator


class featureSelection():
    def __init__(self):
        pass

    def rankFeaturesChi2(self, X, y):
        return np.argsort(SelectKBest(chi2).fit(X, y).scores_)[::-1]

    def rankFeaturesForest(self, X, y, n_estimators):
        return np.argsort(ExtraTreesClassifier(n_estimators=n_estimators).fit(X, y).feature_importances_)[::-1]

    def rankFeaturesFourier(self, X, y):
        sorted_coeffs = sorted(Fourier(X, y).coeff(1).items(), key=operator.itemgetter(1), reverse=True)
        ranked_subsets = []
        for tuple in sorted_coeffs:
            ranked_subsets.append(tuple[0][0])
        return ranked_subsets