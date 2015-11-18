import os

class logger():
    def __init__(self, baseDir="/home/alex/KnowEng/logs/", logFileName=None, shouldLog=True):
        self.shouldLog = shouldLog
        if logFileName:
            self.logFileName = baseDir + logFileName
        else:
            logFiles = os.listdir(baseDir)
            if logFiles:
                for i in range(0, len(logFiles)):
                    try:
                        logFiles[i] = int(logFiles[i].rstrip(".txt"))
                    except:
                        continue
                self.logFileName = baseDir + str(max(logFiles) + 1) + ".txt"
            else:
                self.logFileName = baseDir + "0.txt"

    def log(self, contents, shouldPrint=True):
        if self.shouldLog:
            logFile = open(self.logFileName, 'a')
            logFile.write(contents + '\n')
            logFile.close()
        if shouldPrint:
            print(contents)