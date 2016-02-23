from optparse import OptionParser

parser = OptionParser()
parser.add_option("-f", "--file", dest="filename",
                  help="file with JRip rules", metavar="FILE")
parser.add_option("-o", "--out", dest="outFile",
                  help="file to write rules out to", metavar="FILE")

(options, args) = parser.parse_args()

jRipFile = open(options.filename, 'r')

featureRules = {}

for line in jRipFile:
    line = line.split("=>")[0].split('and')
    for rule in line:
        items = rule.strip("( )").split()
        feature = items[0]
        comp = items[1]
        value = items[2]
        if feature not in featureRules:
            featureRules[feature] = set()

        featureRules[feature].add('{0} {1}'.format(comp, value))

out_fd = open(options.outFile, "w")

for feature, featureRule in featureRules.iteritems():
    for rule in featureRule:
        out_fd.write("{0} {1}\n".format(feature, rule))

