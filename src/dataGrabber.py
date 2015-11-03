class dataGrabber:

    def __init__(self, filtMax=None, filtMin=None):
        self.data = {}
        self.dataIndices = {}
        self.dataHistogram = {}
        self.filtMax = filtMax
        self.filtMin = filtMin

    def getData(self, labelFileName, dataFileName, geneIdx, dataIdx, splitCriteria = None):
        self._getDataHistogram(dataFileName, dataIdx, splitCriteria)
        self._readData(dataFileName, geneIdx, dataIdx, splitCriteria)
        wantedGeneIds = open(labelFileName, "r").read().split()
        labelDict = {}
        for wantedGeneId in wantedGeneIds:
            if wantedGeneId in self.data:
                labelDict[wantedGeneId] = 1
        for geneID in self.data.iterkeys():
            if geneID not in labelDict:
                labelDict[geneID] = 0
        return (self.data, labelDict, self.dataIndices)

    def convertToCSV(self, featureDict, labelDict, fileName):
        csvFile = open(fileName, "w")
        featureLength = len(featureDict.values()[0])
        featureNames = ["class"]
        for i in range(0, featureLength):
            featureNames.append("F{0}".format(i))
        csvFile.write(str(featureNames).strip("[]").replace("'", "") + '\n')
        for key, featureVector in featureDict.iteritems():
            csvFile.write(str(labelDict[key]) + ", " + str(featureVector).strip("[]") + '\n')
        csvFile.close()

    def _readData(self, fileName, geneIdx, dataIdx, splitCriteria):
        dataFile = open(fileName, "r")
        dataDims = self._getDataDimensionality(fileName, dataIdx, splitCriteria)
        for line in dataFile:
            items = line.split(splitCriteria)
            data = items[dataIdx]
            geneId = items[geneIdx]
            if geneId not in self.data:
                self.data[geneId] = [0] * dataDims
            if data in self.dataIndices:
                self.data[geneId][self.dataIndices[data]] = 1
        dataFile.close()

    def _getDataHistogram(self, fileName, dataIdx, splitCriteria):
        dataFile = open(fileName, "r")
        for line in dataFile:
            items = line.split(splitCriteria)
            data = items[dataIdx]
            if data not in self.dataHistogram:
                self.dataHistogram[data] = 0
            self.dataHistogram[data] += 1
        dataFile.close()

    def _getDataDimensionality(self, fileName, dataIdx, splitCriteria):
        dataFile = open(fileName, "r")
        curIndex = 0
        for line in dataFile:
            items = line.split(splitCriteria)
            data = items[dataIdx]
            if data not in self.dataIndices:
                if (self.filtMin == None) or (self.filtMax == None):
                    self.dataIndices[data] = curIndex
                    curIndex = curIndex + 1
                elif (self.dataHistogram[data] >= self.filtMin) and (self.dataHistogram[data] <= self.filtMax):
                    self.dataIndices[data] = curIndex
                    curIndex = curIndex + 1
        dataFile.close()
        return curIndex
