from tabulate import tabulate

class helpers():
    def __init__(self, logger):
        self.logger = logger

    def nicePrintList(self, inList, itemsPerLine=5):
        length = len(inList)
        table = []
        for i in range(0, length, itemsPerLine):
            line = inList[i:i + itemsPerLine]
            if len(line) < itemsPerLine:
                line += ([""] * (itemsPerLine - len(line)))
            table.append(line)
        self.logger.log(tabulate(table) + '\n')

    def getFeaturesFromRipperRules(self, ruleFileName, featureTypes):
        rulesFeatures = set()
        ruleFile = open(ruleFileName, "r")
        for line in ruleFile:
            items = line.split()
            for item in items:
                for featureType in featureTypes:
                    if item[0:len(featureType)] == featureType:
                        rulesFeatures.add(item)
                        break
        rulesFeatures = list(rulesFeatures)

        self.logger.log("Features extracted from: {0}".format(ruleFileName))
        self.nicePrintList(rulesFeatures, 4)

        return rulesFeatures

    def dictOfFeaturesToList(self, featureDict, keys):
        featureList = []
        for key in keys:
            featureList.append(featureDict[key])
        return featureList

    def getLabelSets(self, labelDict):
        positiveSet = set()
        negativeSet = set()
        for k, v in labelDict.iteritems():
            if v == 1:
                positiveSet.add(k)
            else:
                negativeSet.add(k)
        return (positiveSet, negativeSet)

    def getFeaturesByLabel(self, featureDict, labelDict, label):
        features = []
        for k, v in labelDict.iteritems():
            if v == label:
                features.append(featureDict[k])
        return features