import inspect
from tabulate import tabulate


class dataFileDescriptor:
    def __init__(self, fileName, filterRule = lambda feature, count: True, geneIdx = 1, dataIdx = 0, splitCriteria = None):
        self.fileName = fileName
        self.filterRule = filterRule
        self.geneIdx = geneIdx
        self.dataIdx = dataIdx
        self.splitCriteria = None

class dataGrabber:

    def __init__(self, logger):
        self.logger = logger
        self.data = {}
        self.dataIndices = {}
        self.dataHistogram = {}

    def getData(self, positiveLabelsFileName, dataFileDescriptors):
        headers = ["Data type", "File Name", "Filter Rule"]
        table = [["Positive labels", positiveLabelsFileName, "None"]]
        for dataFileDescriptor in dataFileDescriptors:
            table.append(["Features", dataFileDescriptor.fileName, inspect.getsource(dataFileDescriptor.filterRule)])
        self.logger.log(tabulate(table, headers=headers) + '\n')

        self._getDataHistogram(dataFileDescriptors)
        self._readData(dataFileDescriptors)

        wantedGeneIds = open(positiveLabelsFileName, "r").read().split()
        labelDict = {}
        count = 0
        for wantedGeneId in wantedGeneIds:
            if wantedGeneId in self.data:
                labelDict[wantedGeneId] = 1
            if wantedGeneId not in self.data:
                count += 1
        self.logger.log("Missing {0} geneIDs from {1} in data file set\n".format(count, positiveLabelsFileName))
        for geneID in self.data.iterkeys():
            if geneID not in labelDict:
                labelDict[geneID] = 0
        return (self.data, labelDict, self.dataIndices)

    def convertToCSV(self, featureDict, labelDict, fileName):
        csvFile = open(fileName, "w")
        featureNames = ["GeneID"] + [""] * len(self.dataIndices)
        for edgeName, index in self.dataIndices.iteritems():
            featureNames[index + 1] = edgeName
        featureNames.append("Class")
        csvFile.write(str(featureNames).strip("[]").replace("'", "") + '\n')
        for key, featureVector in featureDict.iteritems():
            if labelDict[key]:
                label = "POSITIVE"
            else:
                label = "NEGATIVE"
            if label == "POSITIVE":
                for i in range (0, 5):
                    outString = str(key) + ", " + str(featureVector).strip("[]").replace("1", "TRUE").replace("0", "FALSE") + ", " + str(label)
                    csvFile.write(outString + '\n')
            else:
                outString = str(key) + ", " + str(featureVector).strip("[]").replace("1", "TRUE").replace("0", "FALSE") + ", " + str(label)
                csvFile.write(outString + '\n')
        csvFile.close()

    def _readData(self, dataFileDescriptors):
        dataDims = self._getDataDimensionality(dataFileDescriptors)
        for fileDescriptor in dataFileDescriptors:
            dataFile = open(fileDescriptor.fileName, "r")
            for line in dataFile:
                items = line.split(fileDescriptor.splitCriteria)
                data = items[fileDescriptor.dataIdx]
                geneId = items[fileDescriptor.geneIdx]
                if geneId not in self.data:
                    self.data[geneId] = [0] * dataDims
                if data in self.dataIndices:
                    self.data[geneId][self.dataIndices[data]] = 1
            dataFile.close()

    def _getDataHistogram(self, dataFileDescriptors):
        for fileDescriptor in dataFileDescriptors:
            dataFile = open(fileDescriptor.fileName, "r")
            for line in dataFile:
                items = line.split(fileDescriptor.splitCriteria)
                data = items[fileDescriptor.dataIdx]
                geneId = items[fileDescriptor.geneIdx]
                if data not in self.dataHistogram:
                    self.dataHistogram[data] = set()
                self.dataHistogram[data].add(geneId)
            dataFile.close()
        for dataTypeName, geneSet in self.dataHistogram.iteritems():
            self.dataHistogram[dataTypeName] = len(geneSet)

    def _getDataDimensionality(self, dataFileDescriptors):
        curIndex = 0
        for fileDescriptor in dataFileDescriptors:
            dataFile = open(fileDescriptor.fileName, "r")
            for line in dataFile:
                items = line.split(fileDescriptor.splitCriteria)
                data = items[fileDescriptor.dataIdx]
                if data not in self.dataIndices:
                    if fileDescriptor.filterRule(data, self.dataHistogram[data]):
                        self.dataIndices[data] = curIndex
                        curIndex = curIndex + 1
            dataFile.close()
        return curIndex
