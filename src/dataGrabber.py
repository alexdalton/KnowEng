class dataGrabber:

    def __init__(self):
        self.data = {}
        self.dataIndices = {}

    def getData(self, labelFileName, dataFileName, geneIdx, dataIdx, splitCriteria = None):
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

    def _readData(self, fileName, geneIdx, dataIdx, splitCriteria):
        dataFile = open(fileName, "r")
        dataDims = self._getDataDimensionality(fileName, dataIdx, splitCriteria)
        for line in dataFile:
            items = line.split(splitCriteria)
            data = items[dataIdx]
            geneId = items[geneIdx]
            if geneId not in self.data:
                self.data[geneId] = [0] * dataDims
            self.data[geneId][self.dataIndices[data]] = 1
        dataFile.close()

    def _getDataDimensionality(self, fileName, dataIdx, splitCriteria):
        dataFile = open(fileName, "r")
        curIndex = 0
        for line in dataFile:
            items = line.split(splitCriteria)
            data = items[dataIdx]
            if data not in self.dataIndices:
                self.dataIndices[data] = curIndex
                curIndex = curIndex + 1
        dataFile.close()
        return curIndex

