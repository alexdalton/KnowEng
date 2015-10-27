

class helpers():
    def __init__(self):
        pass

    def dictOfFeaturesToList(self, featureDict, keys):
        featureList = []
        for key in keys:
            featureList.append(featureDict[key])
        return featureList

    def getLabelSets(self, labelDict):
        positiveSet = set()
        unlabeledSet = set()
        for k, v in labelDict.iteritems():
            if v == 1:
                positiveSet.add(k)
            else:
                unlabeledSet.add(k)
        return (positiveSet, unlabeledSet)

    def getFeaturesByLabel(self, featureDict, labelDict, label):
        features = []
        for k, v in labelDict.iteritems():
            if v == label:
                features.append(featureDict[k])
        return features