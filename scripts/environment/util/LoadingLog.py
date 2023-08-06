import re
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

        elapsedTime = str(timedelta(seconds=round(time.time() - self.startTime)))

        # Use regular expressions to extract days, hours, minutes, and seconds from the string
        pattern = r"((?P<days>\d+) days?, )?(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+)"
        match = re.match(pattern, elapsedTime)
        if match:
            days = int(match.group("days")) if match.group("days") else 0
            hours = int(match.group("hours"))
            minutes = int(match.group("minutes"))
            seconds = int(match.group("seconds"))
        else:
            print("Unable to parse elapsed time.")

        # Format the time string in a human-readable format
        timestring = (
                         (str(days) + "d ") if days > 0 else ""
                     ) + (  # Add days if the elapsed time is more than 0 days
                         (str(hours) + "h ") if hours > 0 else ""
                     ) + (  # Add hours if the elapsed time is more than 0 hours
                         (str(minutes) + "m ") if minutes > 0 else ""
                     ) + (  # Add minutes if the elapsed time is more than 0 minutes
                         (str(seconds) + "s") if seconds > 0 else ""
                     )  # Add seconds if the elapsed time is more than 0 seconds

        self.printData = timestring + " |" + str(loadData) + "| " + str(round(percent * 100)) + "%"
        sys.stdout.write("\r" + self.printData)
        sys.stdout.flush()

    def tick(self):
        self.printProgress(self.progress)
