class dataGrabber:

    def __init__(self):
        self.data = {}
        self.dataIndices = {}

    def getData(self, labelFileName, dataFileName, geneIdx, dataIdx, splitCriteria = None):
        self._readData(dataFileName, geneIdx, dataIdx, splitCriteria)
        wantedGeneIds = open(labelFileName, "r").read().split()
        positiveData = {}
        for wantedGeneId in wantedGeneIds:
            if wantedGeneId in self.data:
                positiveData[wantedGeneId] = self.data[wantedGeneId]
                del self.data[wantedGeneId]
        return (positiveData, self.data)

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

