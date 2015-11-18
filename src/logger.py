import os

class logger():
    def __init__(self, logFileName=None):
        if logFileName:
            self.logFileName = logFileName
        else:
            logFiles = os.listdir("/home/alex/KnowEng/logs")
            if logFiles:
                for i in range(0, len(logFiles)):
                    try:
                        logFiles[i] = int(logFiles[i].rstrip(".txt"))
                    except:
                        continue
                self.logFileName = str(max(logFiles) + 1) + ".txt"
            else:
                self.logFileName = "/home/alex/KnowEng/logs/0.txt"

    def log(self, contents):
        logFile = open(self.logFileName, 'a')
        logFile.write(contents + '\n')
        logFile.close()
        print(contents)