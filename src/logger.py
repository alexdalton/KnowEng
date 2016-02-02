import os

class logger():
    def __init__(self, baseDir="/home/alex/KnowEng/logs/", logFileName=None, shouldLog=True):
        self.shouldLog = shouldLog
        if logFileName:
            self.logFileName = os.path.join(baseDir, logFileName)
        else:
            logFiles = os.listdir(baseDir)
            if logFiles:
                for i in range(0, len(logFiles)):
                    try:
                        logFiles[i] = int(logFiles[i].rstrip(".txt"))
                    except:
                        logFiles[i] = 0
                self.logFileName = os.path.join(baseDir, str(max(logFiles) + 1) + ".txt")
            else:
                self.logFileName = os.path.join(baseDir, "0.txt")

    def log(self, contents, shouldPrint=True):
        if self.shouldLog:
            logFile = open(self.logFileName, 'a')
            logFile.write(contents + '\n')
            logFile.close()
        if shouldPrint:
            print(contents)