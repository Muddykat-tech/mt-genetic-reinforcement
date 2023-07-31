import sys
import time
from datetime import timedelta

class PrintLoader:
    def __init__(self, max, symbol):
        self.max = (max - 1)
        self.symbol = symbol
        self.printData = ""
        self.startTime = time.time()
        self.progress = 0

    def reset(self):
        self.startTime = time.time()

    def printProgress(self, progress):
        self.progress = progress
        percent = (progress / self.max)
        maxCharacters = int(185 * percent)
        loadingbar = ["-"] * 185
        for index in range(maxCharacters):
            loadingbar[index] = self.symbol

        loadData = ''.join(loadingbar)

        elapsedTime = str(timedelta(seconds=round(time.time() - self.startTime))).split(':')

        timestring = ((str(int(elapsedTime[0])) + "h ") if int(elapsedTime[0]) > 0 else "") + (
            (str(int(elapsedTime[1])) + "m ") if int(elapsedTime[1]) > 0 else "") + (
                         (str(int(elapsedTime[2])) + "s") if int(elapsedTime[2]) > 0 else "")

        self.printData = timestring + " |" + str(loadData) + "| " + str(round(percent * 100)) + "%"
        sys.stdout.write("\r" + self.printData)
        sys.stdout.flush()

    def tick(self):
        self.printProgress(self.progress)
