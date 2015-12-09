#!/usr/bin/env python

from optparse import OptionParser
from os import path

parser = OptionParser()
parser.add_option("-f", "--file", dest="ruleFileName",
                  help="name of file with ripper output", metavar="FILE")
parser.add_option("-o", "--out", dest="outFileName",
                  help="name of file to write output to", metavar="FILE")

(options, args) = parser.parse_args()
ruleFileName = path.abspath(options.ruleFileName)
outFileName = path.abspath(options.outFileName)

rulesFeatures = set()
ruleFile = open(ruleFileName, "r")
for line in ruleFile:
    items = line.split("=>")[0].split(" and ")
    for item in items:
        feature = item.strip("( )").split("=")[0].strip()
        if feature != '':
            rulesFeatures.add(feature)

ruleFile.close()
rulesFeatures = list(rulesFeatures)

out = open(outFileName, "w")
for feature in rulesFeatures:
    if feature != rulesFeatures[-1]:
        out.write(feature + '\n')
    else:
        out.write(feature)

out.close()