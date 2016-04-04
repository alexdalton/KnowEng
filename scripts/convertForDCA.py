from optparse import OptionParser

parser = OptionParser()
parser.add_option("-f", "--file", dest="filename",
                  help="edge file to convert for DCA", metavar="FILE")

(options, args) = parser.parse_args()

edgeFile = open(options.filename, 'r')

genesFile = open(options.filename + "_genes.txt", "w")
adjacencyFile = open(options.filename + "_adjacency.txt", "w")

geneDict = {}
geneNum = 1

for edge in edgeFile:
    items = edge.split()
    g1 = items[0]
    g2 = items[1]
    weight = items[2]
    if g1 not in geneDict:
        geneDict[g1] = geneNum
        genesFile.write(g1 + '\n')
        geneNum += 1
    if g2 not in geneDict:
        geneDict[g2] = geneNum
        genesFile.write(g2 + '\n')
        geneNum += 1
    adjacencyFile.write("{0}\t{1}\t{2}\n".format(geneDict[g1], geneDict[g2], weight))
