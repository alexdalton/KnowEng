from dataGrabber import dataGrabber
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from fourier import fourier
import operator
from PEBL import PEBL
import numpy as np

labelFile = "/home/alex/KnowEng/data/LEE_LIVER_CANCER_ACOX1.CB.txt"
dataFile = "/home/alex/KnowEng/data/ENSG.kegg_pathway.txt"

dataRetriever = dataGrabber()
(featureVectorDict, labelDict, dataIndices) = dataRetriever.getData(labelFile, dataFile, 1, 0)

def _getLabelSets():
    positiveSet = set()
    negativeSet = set()
    for k, v in labelDict.iteritems():
        if v == 1:
            positiveSet.add(k)
        else:
            negativeSet.add(k)
    return (positiveSet, negativeSet)


features = []
labels = []

for id in labelDict.iterkeys():
    features.append(featureVectorDict[id])
    labels.append(labelDict[id])

# X_new = SelectKBest(chi2, k=50).fit_transform(features, labels)
#
# print X_new.shape

# clf = ExtraTreesClassifier()
#
# features_new = clf.fit(features, labels).transform(features)
# importances = clf.feature_importances_
#
# indices = np.argsort(importances)[::-1]
#
# for i in list(indices)[0:10]:
#     for k, v in dataIndices.iteritems():
#         if v == i:
#             print k

# sel = VarianceThreshold(threshold=.95 * (1 - .95))
# y = sel.fit_transform(features)
#
# print len(y[0])

#x = fourier(featureVectorDict, labelDict, len(featureVectorDict.values()[0]))

#print(sorted(x.coeff(2, .20, .20).items(), key=operator.itemgetter(1), reverse=True)[0:10])


#print y
classifierPEBL = PEBL()
trainedSVM = classifierPEBL.train(featureVectorDict, labelDict)

