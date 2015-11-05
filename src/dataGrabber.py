class dataFileDescriptor:
    def __init__(self, fileName, filterRule = lambda x: True, geneIdx = 1, dataIdx = 0, splitCriteria = None):
        self.fileName = fileName
        self.filterRule = filterRule
        self.geneIdx = geneIdx
        self.dataIdx = dataIdx
        self.splitCriteria = None

class dataGrabber:

    def __init__(self):
        self.data = {}
        self.dataIndices = {}
        self.dataHistogram = {}

    def getData(self, positiveLabelsFileName, dataFileDescriptors):
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
        print("Missing {0} geneIDs from {1} in data file set".format(count, positiveLabelsFileName))
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
        x = ["go_small_molecule_metabolic_process",
             "go_glucose_metabolic_process",
             "go_negative_regulation_of_cell_grow",
             "go_protein_homodimerization_activit",
             "go_liver_development",
             "go_enzyme_binding",
             "go_mitochondrion",
             "go_identical_protein_binding",
             "go_negative_regulation_of_neuron_ap",
             "go_autophagosome",
             "go_cellular_nitrogen_compound_metab",
             "go_electron_carrier_activity",
             "go_mitochondrial_matrix",
             "go_iron_ion_binding",
             "go_cell_proliferation",
             "go_condensed_chromosome_kinetochore",
             "go_perinuclear_region_of_cytoplasm"]
        for fileDescriptor in dataFileDescriptors:
            dataFile = open(fileDescriptor.fileName, "r")
            for line in dataFile:
                items = line.split(fileDescriptor.splitCriteria)
                data = items[fileDescriptor.dataIdx]
                if data not in self.dataIndices:
                    #if fileDescriptor.filterRule(self.dataHistogram[data]):
                    if data in x:
                        self.dataIndices[data] = curIndex
                        curIndex = curIndex + 1
            dataFile.close()
        return curIndex
